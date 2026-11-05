"""
Three follow-up improvements to the diffusion-anomaly-gate idea, on the user's own
(no-colleague-engine) simplified analog model, all CPCV-validated:

A) Deterministic full-ELBO atypicality score: average denoising loss over ALL 50 timesteps
   (one noise draw each), instead of 32 random (t, noise) draws -- lower variance, a more
   complete Monte Carlo estimate of the ELBO.
B) Diffusion Time Estimation (DTE)-style score: fit a per-point "best" timestep by finding
   which noise level minimizes reconstruction error, and use that AS the atypicality score
   (not the loss itself) -- a different summary statistic of the same underlying posterior
   p(t|x), per the DTE anomaly-detection literature.
C) Stack: does selecting analogs via the diffusion-score metric (finding 18) AND gating by
   diffusion-anomaly (finding 26) compound, since both come from the same trained denoiser?

Tested on 5 assets (Brent, S&P 500, Bitcoin, Ethereum, Gold). CPCV-confirmed results:

| asset | A) full ELBO (low/high) | C) metric-selection + ELBO gate (low/high) |
|---|---|---|
| Brent | 80.0% / 70.8% (REVERSED) | 67.9% / 55.0% (REVERSED) |
| S&P 500 | 61.3% / 71.4% | 70.0% / 71.4% (flat) |
| Bitcoin | 61.3% / 75.0% | 68.3% / **83.3%** |
| Ethereum | 57.1% / 71.4% | 56.3% / **80.0%** |
| Gold | 57.1% / 66.7% | 59.2% / **75.0%** |
| **method B (DTE-style, min loss)** | weak/flat on every asset, not pursued further |

Finding: stack C is strong and consistent on Bitcoin, Ethereum, and Gold (75-83% on the
high-atypicality bucket), flat on S&P 500, and REVERSES specifically on Brent crude. The
original finding 26 number for Brent (83.3%, from a 32-random-sample loss estimate) does not
survive the lower-variance deterministic ELBO estimator -- that specific number was likely
partly noise, not a robust effect. The refined, honest claim: this combination generalizes well
to crypto-like and metal markets, but not to Brent crude, plausibly because Brent's moves are
more often driven by discrete geopolitical/supply news than by a continuous statistical
structure a diffusion model trained only on price history can capture.
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


class ScoreNet(nn.Module):
    def __init__(self, in_dim=M, hidden=HIDDEN):
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
            loss = nn.functional.mse_loss(net(x_t, t_frac), noise)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, (betas, alphas, alpha_bars)


@torch.no_grad()
def full_elbo_score(net, schedule, point, seed=0):
    """A) deterministic sweep over all N_TIMESTEPS, one noise draw each -- the per-timestep
    losses ARE the (unnormalized) ELBO terms; their mean is a lower-variance atypicality score
    than a 32-sample random subset."""
    betas, alphas, alpha_bars = schedule
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.arange(N_TIMESTEPS, device=device)
    x0 = torch.tensor(point, dtype=torch.float32, device=device).unsqueeze(0).repeat(N_TIMESTEPS, 1)
    noise = torch.randn(x0.shape, generator=g, device=device)
    ab = alpha_bars[t].unsqueeze(-1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    t_frac = (t.float() / N_TIMESTEPS).unsqueeze(-1)
    pred = net(x_t, t_frac)
    per_t_loss = ((pred - noise) ** 2).mean(dim=-1)  # (N_TIMESTEPS,)
    return float(per_t_loss.mean().item()), per_t_loss.cpu().numpy()


def dte_score(per_t_loss):
    """B) Diffusion-Time-Estimation-style summary: the MINIMUM per-timestep loss is the level
    at which the model is most confident this point is "normal" -- a point whose best-case
    loss is still high is atypical in a different sense than one whose average loss is high."""
    return float(per_t_loss.min())


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

    net, schedule, rows = None, None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
        if net is None or k % retrain_every == 0:
            net, schedule = train_scorenet(pc[:j+1], epochs=epochs)

        d_pos = np.sum((pc[:j+1] - pc[j]) ** 2, axis=1)
        idx_euclid = select_analogs(d_pos, j, last_ok)

        feats = net.forward(torch.tensor(pc[:j+1], dtype=torch.float32, device=device),
                             torch.full((j+1, 1), 3 / N_TIMESTEPS, device=device),
                             return_feature=True).detach().cpu().numpy()
        d_diff = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_diff = select_analogs(d_diff, j, last_ok)

        if len(idx_euclid) < 3 or len(idx_diff) < 3:
            continue

        full_elbo, per_t = full_elbo_score(net, schedule, pc[j], seed=k)
        dte = dte_score(per_t)

        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]

        def mean_pred(idx_list):
            rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                    for idx in idx_list if idx + (M-1)*TAU + HORIZON < len(prices)]
            return current_price * np.exp(np.mean(rets)) if rets else None

        euclid_pred, diff_pred = mean_pred(idx_euclid), mean_pred(idx_diff)
        if euclid_pred is None or diff_pred is None:
            continue
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price,
                      "euclid_pred": euclid_pred, "diff_pred": diff_pred,
                      "full_elbo": full_elbo, "dte": dte})
    return pd.DataFrame(rows)


def causal_bucket(df, col, pred_col, min_train=20):
    vals = df[col].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        bucket[i] = "high" if vals[i] >= np.median(vals[:i]) else "low"
    out = df.copy(); out["bucket"] = bucket
    rows = {}
    for b in ("high", "low"):
        sub = out[out.bucket == b]
        if sub.empty:
            continue
        up = sub[pred_col] > sub["current_price"]
        rows[b] = {"dir_acc": float((up == sub["actual_up"]).mean()), "n": len(sub)}
    return rows


def cpcv_bucketed(df, col, pred_col, n_blocks=8, k_test=2, min_train=20):
    vals = df[col].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        bucket[i] = "high" if vals[i] >= np.median(vals[:i]) else "low"
    work = df.iloc[min_train:].copy()
    work["bucket"] = bucket[min_train:]
    n = len(work)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[b], edges[b+1]) for b in range(n_blocks)]
    out = {"low": [], "high": []}
    for combo in itertools.combinations(range(n_blocks), k_test):
        idx = np.concatenate([blocks[b] for b in combo])
        sub = work.iloc[idx]
        for b in ("low", "high"):
            bsub = sub[sub.bucket == b]
            if len(bsub) < 4:
                continue
            up = bsub[pred_col] > bsub["current_price"]
            out[b].append(float((up == bsub["actual_up"]).mean()))
    return out


for ticker, name in [("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin"),
                     ("ETH-USD", "Ethereum"), ("GC=F", "Gold")]:
    t0 = time.time()
    df = collect(ticker, name)
    print(f"\n=== {name} (n={len(df)}, {time.time()-t0:.0f}s) ===")

    for score_col, label in [("full_elbo", "A) full deterministic ELBO"), ("dte", "B) DTE-style (min loss)")]:
        c = cpcv_bucketed(df, score_col, "euclid_pred")
        lo, hi = c["low"], c["high"]
        print(f"  {label:32s} low median={np.median(lo):.3f} (n={len(lo)})  "
              f"high median={np.median(hi):.3f} (n={len(hi)})")

    # C) stack: select analogs via diffusion metric, THEN gate by full-ELBO atypicality
    c_stack = cpcv_bucketed(df, "full_elbo", "diff_pred")
    lo_s, hi_s = c_stack["low"], c_stack["high"]
    print(f"  {'C) diffusion-metric selection + ELBO gate':32s} low median={np.median(lo_s):.3f} (n={len(lo_s)})  "
          f"high median={np.median(hi_s):.3f} (n={len(hi_s)})")
