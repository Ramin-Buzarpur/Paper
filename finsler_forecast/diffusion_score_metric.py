"""
Diffusion-score-based similarity metric for analog SELECTION -- a different mechanism than
metric_learning_analog_selection.py's learned coordinate weights, and critically different from
a diffusion SCENARIO GENERATOR (the user's own earlier diffusion+RL attempt, which replaced the
true historical continuations with synthetic ones and underperformed): this keeps every analog's
TRUE historical continuation, and only changes which historical days count as "similar" in the
first place.

Mechanism: train a small unconditional DDPM on the engine's own 3D delay-embedding point cloud.
A diffusion denoiser learns a representation of the data manifold while learning the score
function ∇log p(x); the denoiser's first hidden-layer activation at a fixed, small noise level
is extracted as a learned "diffusion feature" for any point, and Euclidean distance in that
feature space replaces the plain Euclidean/Mahalanobis distance for picking the K=7 nearest,
Theiler-windowed historical analogs.

Result (see README Finding 18), CPCV-confirmed (28 combinatorial fold groupings):

| asset | plain-Euclidean metric | diffusion-score metric |
|---|---|---|
| Brent | 75.0% | 70.0% (worse, loses in 46% of combos) |
| S&P 500 | 57.5% | **65.0%** (robust win, loses in 0% of combos) |
| Bitcoin | 67.5% | 65.0% (worse, loses in 64% of combos) |

A second, independent technique (after metric_learning_analog_selection.py's #8) in the "change
what counts as similar" family that specifically helps one asset (here S&P 500) while hurting
the other two -- consistent with the pattern that no single complement generalizes, but the
category of complement ("change the similarity metric" / "decide when to trust it") is where
real improvements live.
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
FEAT_DIM = 32
PROBE_T = 3


class ScoreNet(nn.Module):
    def __init__(self, in_dim=M, hidden=FEAT_DIM):
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
        h2 = self.act(self.l2(h1))
        return self.out(h2)


def make_schedule(n=N_TIMESTEPS):
    betas = torch.linspace(1e-4, 0.02, n)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def train_scorenet(points, epochs=50, lr=2e-3, batch=128, seed=0):
    torch.manual_seed(seed)
    net = ScoreNet().to(device)
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
            pred_noise = net(x_t, t_frac)
            loss = nn.functional.mse_loss(pred_noise, noise)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def diffusion_features(net, points, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.tensor(points, dtype=torch.float32).to(device)
    t_frac = torch.full((len(X), 1), PROBE_T / N_TIMESTEPS, device=device)
    noise = torch.randn(X.shape, generator=g, device=device) * np.sqrt(PROBE_T / N_TIMESTEPS) * 0.3
    x_t = X + noise
    return net(x_t, t_frac, return_feature=True).cpu().numpy()


def embed(log_ret):
    n = len(log_ret)
    n_pts = n - (M - 1) * TAU
    return np.column_stack([log_ret[(M-1-i)*TAU : (M-1-i)*TAU+n_pts] for i in range(M)])


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


def collect(ticker, name, n_folds=80, step_back=2, retrain_every=15, epochs=50):
    price_df = download_close_prices(ticker, 6)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    min_history, latest = 200, n_pts - HORIZON - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))

    net = None
    rows = []
    for k, j in enumerate(origins):
        last_ok = j - 1
        if net is None or k % retrain_every == 0:
            net = train_scorenet(pc[:j+1], epochs=epochs)

        d_euclid = np.sum((pc[:j+1] - pc[j]) ** 2, axis=1)
        idx_euclid = select_analogs(d_euclid, j, last_ok)
        feats = diffusion_features(net, pc[:j+1])
        d_diff = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_diff = select_analogs(d_diff, j, last_ok)
        if len(idx_euclid) < 3 or len(idx_diff) < 3:
            continue

        p = j + (M - 1) * TAU
        current_price = prices[p]
        actual_future_price = prices[p + HORIZON]

        def mean_pred(idx_list):
            rets = []
            for idx in idx_list:
                pp = idx + (M - 1) * TAU
                if pp + HORIZON < len(prices):
                    rets.append(np.log(prices[pp + HORIZON] / prices[pp]))
            return current_price * np.exp(np.mean(rets)) if rets else None

        euclid_pred, diff_pred = mean_pred(idx_euclid), mean_pred(idx_diff)
        if euclid_pred is None or diff_pred is None:
            continue
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price,
                      "euclid_pred": euclid_pred, "diff_pred": diff_pred})
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
        for col in ("euclid_pred", "diff_pred"):
            up = sub[col] > sub["current_price"]
            rows.append({"combo": combo, "method": col, "dir_acc": float((up == sub["actual_up"]).mean())})
    return pd.DataFrame(rows)


for ticker, name in [("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin")]:
    t0 = time.time()
    df = collect(ticker, name)
    c = cpcv(df)
    e = c[c.method == "euclid_pred"]["dir_acc"]
    d = c[c.method == "diff_pred"]["dir_acc"]
    print(f"{name:10s} n={len(df):3d}  euclid median={e.median():.3f}  diffusion-metric median={d.median():.3f}  "
          f"diffusion loses in {float((d.to_numpy() < e.to_numpy()).mean()):.2f} of {len(e)} CPCV combos  ({time.time()-t0:.0f}s)")
