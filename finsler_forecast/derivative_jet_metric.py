"""
Partial-derivative-based similarity metric for analog selection -- without a network (closed-
form) and with one (learned), per the user's question.

Mechanism: extend each embedded state x_j = [r(t), r(t-tau), r(t-2tau)] with its local first
derivative (velocity, v_j = x_{j+1} - x_j, already present in geometry.py as
tangent_velocities) and local second derivative (acceleration, a_j = v_{j+1} - v_j, a genuinely
new feature -- the base method never uses curvature of the trajectory). Two ways to use
[x, v, a] as a similarity space:

  A) NO NETWORK: closed-form distance = ||x_i - x_j||^2 + w_v*||v_i - v_j||^2 + w_a*||a_i -
     a_j||^2, fixed equal weights -- a pure jet-space (position+velocity+acceleration) metric,
     zero training.
  B) WITH NETWORK: train the same diffusion-denoiser-as-metric-learner (finding 18) on the
     9-dim extended [x, v, a] state instead of just the 3-dim position, and use its learned
     feature as the metric -- lets the network find a nonlinear combination instead of the
     fixed equal weights in (A).

Both tested against the plain position-only Euclidean baseline, CPCV-confirmed.

CAUGHT AND FIXED DURING DEVELOPMENT: the first version of this script computed v/a with
FORWARD differences (v_j = x_{j+1} - x_j). For the query/origin point itself this leaks
tomorrow's return into today's similarity feature -- it produced an implausible ~90% accuracy
across all three assets, 0% CPCV loss everywhere, which is exactly the kind of result this
whole investigation's discipline exists to catch rather than report. `jet_features` below uses
BACKWARD differences instead (v_j = x_j - x_{j-1}), causal by construction for every point,
origin included.

Result after the fix (see README Finding 24): no improvement over what's already known.

| asset | position-only | +derivatives (no network) | +derivatives (network) |
|---|---|---|---|
| Brent | 75.0% | 65.0% (worse, loses 57%) | 75.0% (tie, loses 39%) |
| S&P 500 | 57.5% | 60.0% (mild, loses 11%) | 65.0% (loses 25% -- same level as finding 18's plain diffusion metric, not better) |
| Bitcoin | 67.5% | 65.0% (worse, loses 50%) | 65.0% (worse, loses 61%) |

Adding local curvature (acceleration) to the similarity space, with or without a network to
combine it, doesn't add information beyond what's already captured by position + the
network's own learned nonlinearity in finding 18. Reported in full, including the caught bug,
because the methodology point (catch implausibly good results before trusting them) matters as
much as the numeric outcome.
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from data import download_close_prices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M, TAU, K, THEILER, HORIZON = 3, 1, 7, 10, 3
N_TIMESTEPS = 50
HIDDEN = 32


def embed(log_ret):
    n = len(log_ret)
    n_pts = n - (M - 1) * TAU
    return np.column_stack([log_ret[(M-1-i)*TAU : (M-1-i)*TAU+n_pts] for i in range(M)])


def jet_features(pc):
    """[x, v, a] per point: position, BACKWARD velocity, BACKWARD acceleration -- causal by
    construction, uses only data up to and including index j (v_j = x_j - x_{j-1}, a_j = v_j -
    v_{j-1} = x_j - 2x_{j-1} + x_{j-2}). First 2 rows have undefined v/a (no past data yet).
    (An earlier version used forward differences, which for the ORIGIN point leaked tomorrow's
    return into today's similarity feature -- caught via an implausible ~90% accuracy result.)"""
    n = len(pc)
    v = np.zeros_like(pc)
    a = np.zeros_like(pc)
    v[1:] = pc[1:] - pc[:-1]
    a[2:] = v[2:] - v[1:-1]
    return np.concatenate([pc, v, a], axis=1)  # (n, 9)


def select_analogs(dist, origin_idx, last_ok):
    order = np.argsort(dist)
    chosen = []
    for j in order:
        if j > last_ok or j == origin_idx:
            continue
        if all(abs(int(j) - c) > THEILER for c in chosen):
            chosen.append(int(j))
        if len(chosen) == K:
            break
    return chosen


class ScoreNet9(nn.Module):
    def __init__(self, in_dim=9, hidden=HIDDEN):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.l1 = nn.Linear(in_dim + hidden, hidden)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x_t, t_frac, return_feature=False):
        te = self.time_embed(t_frac)
        h1 = self.act(self.l1(torch.cat([x_t, te], dim=-1)))
        if return_feature:
            return h1
        return self.out(self.act(self.l2(h1)))


def make_schedule(n=N_TIMESTEPS):
    betas = torch.linspace(1e-4, 0.02, n)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


def train_scorenet9(points, epochs=50, lr=2e-3, batch=128, seed=0):
    torch.manual_seed(seed)
    net = ScoreNet9().to(device)
    betas, alphas, alpha_bars = [t.to(device) for t in make_schedule()]
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    X = torch.tensor(points, dtype=torch.float32).to(device)
    n = len(X)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x0 = X[idx]
            t = torch.randint(0, N_TIMESTEPS, (len(idx),), device=device)
            noise = torch.randn_like(x0)
            ab = alpha_bars[t].unsqueeze(-1)
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
            t_frac = (t.float() / N_TIMESTEPS).unsqueeze(-1)
            loss = nn.functional.mse_loss(net(x_t, t_frac), noise)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def net9_features(net, points, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.tensor(points, dtype=torch.float32).to(device)
    t_frac = torch.full((len(X), 1), 3 / N_TIMESTEPS, device=device)
    noise = torch.randn(X.shape, generator=g, device=device) * np.sqrt(3 / N_TIMESTEPS) * 0.3
    return net(X + noise, t_frac, return_feature=True).cpu().numpy()


def collect(ticker, name, n_folds=80, step_back=2, retrain_every=15, epochs=50,
            w_v=1.0, w_a=1.0):
    price_df = download_close_prices(ticker, 6)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    jet = jet_features(pc)   # (n, 9) = [x(3), v(3), a(3)]
    weights = np.array([1, 1, 1, w_v, w_v, w_v, w_a, w_a, w_a], dtype=np.float64)

    n_pts = len(pc)
    min_history, latest = 200, n_pts - HORIZON - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))

    net, rows = None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
        if net is None or k % retrain_every == 0:
            net = train_scorenet9(jet[:j+1], epochs=epochs)

        # baseline: plain position-only Euclidean
        d_pos = np.sum((pc[:j+1] - pc[j]) ** 2, axis=1)
        idx_pos = select_analogs(d_pos, j, last_ok)

        # A) no-network: closed-form jet-space distance with fixed weights
        d_jet = np.sum(((jet[:j+1] - jet[j]) * weights) ** 2, axis=1)
        idx_jet = select_analogs(d_jet, j, last_ok)

        # B) with network: learned feature from a diffusion model trained on the 9-dim jet space
        feats = net9_features(net, jet[:j+1])
        d_net = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_net = select_analogs(d_net, j, last_ok)

        if min(len(idx_pos), len(idx_jet), len(idx_net)) < 3:
            continue

        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]

        def mean_pred(idx_list):
            rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                    for idx in idx_list if idx + (M-1)*TAU + HORIZON < len(prices)]
            return current_price * np.exp(np.mean(rets)) if rets else None

        pos_pred, jet_pred, net_pred = mean_pred(idx_pos), mean_pred(idx_jet), mean_pred(idx_net)
        if None in (pos_pred, jet_pred, net_pred):
            continue
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price,
                      "pos_pred": pos_pred, "jet_pred": jet_pred, "net_pred": net_pred})
    return pd.DataFrame(rows)


def cpcv(df, n_blocks=8, k_test=2):
    n = len(df)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[b], edges[b+1]) for b in range(n_blocks)]
    rows = []
    for combo in itertools.combinations(range(n_blocks), k_test):
        idx = np.concatenate([blocks[b] for b in combo])
        sub = df.iloc[idx]
        if len(sub) < 5:
            continue
        for col in ("pos_pred", "jet_pred", "net_pred"):
            up = sub[col] > sub["current_price"]
            rows.append({"combo": combo, "method": col, "dir_acc": float((up == sub["actual_up"]).mean())})
    return pd.DataFrame(rows)


for ticker, name in [("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin")]:
    t0 = time.time()
    df = collect(ticker, name)
    c = cpcv(df)
    p = c[c.method == "pos_pred"]["dir_acc"].to_numpy()
    j = c[c.method == "jet_pred"]["dir_acc"].to_numpy()
    nn_ = c[c.method == "net_pred"]["dir_acc"].to_numpy()
    print(f"{name:10s} n={len(df):3d}  position-only={np.median(p):.3f}  "
          f"+derivatives(no-network)={np.median(j):.3f} (loses {float((j<p).mean()):.2f})  "
          f"+derivatives(network)={np.median(nn_):.3f} (loses {float((nn_<p).mean()):.2f})  "
          f"({time.time()-t0:.0f}s)")
