"""
Corrected test: finding 26's diffusion-reconstruction-error gate, applied to the REAL
karcher_pred from the colleague's actual FinslerRandersGeodesicEngine (via benchmark.py's
collect_folds, which uses the real Mahalanobis/Ledoit-Wolf local metric) -- not the simplified
plain-Euclidean reimplementation that findings 18-26 actually used. This checks whether the
gate's effect survives when applied to the real engine's output, or whether it was an artifact
of the simplified baseline.

IMPORTANT CONTEXT: this correction applies to every "change similarity" finding from #18
onward (diffusion score metric, VAE, KAN, Transformer-MAE, metric learning, derivatives,
fracdiff, the diffusion-anomaly gate) -- none of them call FinslerRandersGeodesicEngine; they
all reimplement a simplified plain-Euclidean version of the analog-selection idea and compare
against that simplified baseline, not the real engine's Mahalanobis-metric karcher. This script
is the first to re-test one of those findings (#26, the strongest) against the real engine.

Result (see README Finding 27): the effect does NOT survive intact.

| asset | low_typical (real karcher) | high_atypical (real karcher) | vs. finding 26 (simplified) |
|---|---|---|---|
| Brent | 54.5% | 75.0% | same direction, weaker (was 70.7% / 83.3%) |
| S&P 500 | 63.1% | 33.3% | REVERSED (was 50.0% / 60.0%) |
| Bitcoin | 47.7% | 57.1% | same direction, weaker (was 61.3% / 80.0%) |

The gate's direction survives on Brent and Bitcoin, diminished, but completely reverses on S&P
500. Finding 26, as originally stated, overstated its own strength and universality: it was
real for the simplified reimplementation, but partly an artifact of testing against a weaker
baseline (plain Euclidean) rather than the colleague's actual local-covariance metric. This is
a direct, load-bearing correction to the investigation's headline result, caught only because
the user asked whether the "best" results actually touched the colleague's real method.
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from benchmark import download_prices, collect_folds  # uses the REAL FinslerRandersGeodesicEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M, TAU = 3, 1
N_TIMESTEPS = 50
HIDDEN = 32
N_LOSS_SAMPLES = 32


class ScoreNet(nn.Module):
    def __init__(self, in_dim=M, hidden=HIDDEN):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.l1 = nn.Linear(in_dim + hidden, hidden)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x_t, t_frac):
        te = self.time_embed(t_frac)
        h1 = self.act(self.l1(torch.cat([x_t, te], dim=-1)))
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
def atypicality_score(net, schedule, point, seed=0):
    betas, alphas, alpha_bars = schedule
    g = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.tensor(point, dtype=torch.float32, device=device).unsqueeze(0).repeat(N_LOSS_SAMPLES, 1)
    t = torch.randint(0, N_TIMESTEPS, (N_LOSS_SAMPLES,), generator=g, device=device)
    noise = torch.randn(x0.shape, generator=g, device=device)
    ab = alpha_bars[t].unsqueeze(-1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    t_frac = (t.float() / N_TIMESTEPS).unsqueeze(-1)
    pred = net(x_t, t_frac)
    return float(((pred - noise) ** 2).mean().item())


def embed_point_cloud(log_ret):
    n = len(log_ret)
    n_pts = n - (M - 1) * TAU
    return np.column_stack([log_ret[(M-1-i)*TAU : (M-1-i)*TAU+n_pts] for i in range(M)])


def run(ticker, name, freq, horizon=3, n_folds=80, step_back=2, retrain_every=15, epochs=50):
    t0 = time.time()
    price_df = download_prices(ticker, "daily")
    prices_full = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices_full))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed_point_cloud(log_ret) / scale

    # REAL karcher_pred, from the actual FinslerRandersGeodesicEngine
    folds = collect_folds(price_df, ticker, name, horizon=horizon, n_folds=n_folds,
                           step_back=step_back, freq=freq)
    if folds.empty:
        print(f"{name}: no usable folds from the real engine")
        return

    net, schedule = None, None
    rows = []
    for k, row in folds.reset_index(drop=True).iterrows():
        origin_date = row["origin_date"]
        origin_pos = price_df.index.get_loc(origin_date)  # index into prices_full
        j = origin_pos - (M - 1) * TAU  # corresponding index into the embedded point cloud `pc`
        if j < 0 or j >= len(pc):
            continue
        if net is None or k % retrain_every == 0:
            net, schedule = train_scorenet(pc[: max(j, 50)], epochs=epochs)
        atyp = atypicality_score(net, schedule, pc[j], seed=int(k))
        rows.append({
            "current_price": row["current_price"], "actual_future_price": row["actual_future_price"],
            "actual_up": row["actual_up"], "karcher_pred": row["karcher_pred"], "atypicality": atyp,
        })

    df = pd.DataFrame(rows)
    if len(df) < 20:
        print(f"{name}: too few matched folds ({len(df)})")
        return

    # causal bucket + CPCV, same discipline as the original finding 26
    vals = df["atypicality"].to_numpy()
    min_train = 20
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        bucket[i] = "high_atypical" if vals[i] >= np.median(vals[:i]) else "low_typical"
    work = df.iloc[min_train:].copy()
    work["bucket"] = bucket[min_train:]
    n = len(work)
    n_blocks, k_test = 8, 2
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[b], edges[b+1]) for b in range(n_blocks)]
    out = {"low_typical": [], "high_atypical": []}
    for combo in itertools.combinations(range(n_blocks), k_test):
        idx = np.concatenate([blocks[b] for b in combo])
        sub = work.iloc[idx]
        for b in ("low_typical", "high_atypical"):
            bsub = sub[sub.bucket == b]
            if len(bsub) < 4:
                continue
            up = bsub["karcher_pred"] > bsub["current_price"]
            out[b].append(float((up == bsub["actual_up"]).mean()))

    lo, hi = out["low_typical"], out["high_atypical"]
    print(f"{name:10s} n_folds={len(df):3d}  REAL-engine karcher gated by diffusion atypicality  "
          f"low_typical median={np.median(lo):.3f} (n={len(lo)})  "
          f"high_atypical median={np.median(hi):.3f} (n={len(hi)})  ({time.time()-t0:.0f}s)")


for ticker, name, freq in [("BZ=F", "Brent", "B"), ("^GSPC", "S&P500", "B"), ("BTC-USD", "Bitcoin", "D")]:
    run(ticker, name, freq)
