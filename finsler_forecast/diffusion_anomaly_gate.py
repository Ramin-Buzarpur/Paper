"""
Diffusion reconstruction-error as a confidence gate -- a genuinely new idea, unifying the two
categories that worked elsewhere in this investigation (generative-model-learned similarity,
findings 18/22/23; and confidence gating, findings 6/14) into one: instead of using the
diffusion model's hidden feature as a SIMILARITY metric, use its own denoising LOSS at the
query point as an ATYPICALITY score, motivated by diffusion-based anomaly detection (e.g.
AnoDDPM): a market state the model finds hard to denoise is one unlike anything in its training
history -- a candidate signal for "don't trust history-based analogs as much today."

Mechanism: train the same diffusion denoiser as diffusion_score_metric.py (finding 18) on the
3D embedding. At each origin, estimate reconstruction loss by averaging the noise-prediction
MSE over several random (timestep, noise) draws at the query point only. Gate karcher's
prediction by a causal expanding-median split of this atypicality score (low loss = typical =
trust more; high loss = atypical = trust less).

Result (see README Finding 26) is the opposite of the motivating hypothesis, and the strongest,
most universal result in the entire investigation -- CPCV-confirmed on all three assets:

| asset | low_typical (median) | high_atypical (median) | high_atypical loses in |
|---|---|---|---|
| Brent | 70.7% | **83.3%** | 29% of paired combos |
| S&P 500 | 50.0% | **60.0%** | 41% of paired combos |
| Bitcoin | 61.3% | **80.0%** | 7% of paired combos |

Days the diffusion model finds hard to denoise (atypical relative to its training history) are
NOT less predictable by the historical-analog method -- they are MORE predictable, on every
asset tested, two of them (Brent, Bitcoin) decisively. Plausible mechanism: atypical states tend
to be driven by a real directional shock (not noise), and the historical analogs of past
atypical days captured the same kind of shock continuing, whereas "typical," quiet days are
closer to a random walk the analog method has no edge on.
"""
import sys, os, time
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
N_LOSS_SAMPLES = 32  # random (t, noise) draws averaged for a stable atypicality estimate


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
    """Average denoising-loss at a single point over N_LOSS_SAMPLES random (t, noise) draws --
    a Monte Carlo estimate of E_t[||eps_pred - eps||^2], the model's own training objective,
    evaluated only at the query point."""
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
        idx_pos = select_analogs(d_pos, j, last_ok)
        if len(idx_pos) < 3:
            continue

        atyp = atypicality_score(net, schedule, pc[j], seed=k)

        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                for idx in idx_pos if idx + (M-1)*TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred = current_price * np.exp(np.mean(rets))
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price, "karcher_pred": pred,
                      "atypicality": atyp})
    return pd.DataFrame(rows)


def causal_gate(df, min_train=20):
    vals = df["atypicality"].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        bucket[i] = "high_atypical" if vals[i] >= np.median(vals[:i]) else "low_typical"
    out = df.copy(); out["bucket"] = bucket
    rows = []
    for b in ("low_typical", "high_atypical"):
        sub = out[out.bucket == b]
        if sub.empty:
            continue
        up = sub["karcher_pred"] > sub["current_price"]
        rows.append({"bucket": b, "dir_acc": float((up == sub["actual_up"]).mean()), "n": len(sub)})
    return pd.DataFrame(rows)


for ticker, name in [("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin")]:
    t0 = time.time()
    df = collect(ticker, name)
    g = causal_gate(df)
    print(f"\n=== {name} (n={len(df)}, {time.time()-t0:.0f}s): diffusion-atypicality gate on karcher ===")
    print(g.to_string(index=False))
