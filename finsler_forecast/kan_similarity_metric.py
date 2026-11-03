"""
Kolmogorov-Arnold Network (KAN) as the diffusion denoiser's backbone -- a genuinely uncommon
architecture (2024, Liu et al., "KAN: Kolmogorov-Arnold Networks"): instead of a fixed
nonlinearity applied after a linear combination (standard MLP), each edge carries its own
learnable 1D function, summed at each node. This is a SIMPLIFIED KAN layer (each edge's
univariate function is a small learnable combination of Fourier basis functions, not the
official B-spline implementation) -- flagged explicitly, not presented as the canonical KAN.

Same diffusion-score-as-similarity-metric mechanism as diffusion_score_metric.py (finding 18),
swapping only the denoiser's internal architecture, to see whether a KAN backbone extracts a
better (or worse) similarity feature than the plain MLP version.

Result (see README Finding 23), CPCV-confirmed: nearly identical to the plain-MLP diffusion
metric on every asset -- the S&P 500 win is architecture-independent, not an artifact of the
specific MLP used in finding 18:

| asset | plain-Euclidean | KAN-metric | KAN loses in |
|---|---|---|---|
| Brent | 75.0% | 70.0% | 46% of combos |
| S&P 500 | 57.5% | 65.0% | 7% of combos (even lower than MLP's 0%... within noise of it) |
| Bitcoin | 67.5% | 65.0% | 75% of combos |

The takeaway is about the IDEA, not the architecture: learning a similarity metric from a
generative/denoising objective helps S&P 500 regardless of whether the denoiser is an MLP or a
KAN; a self-supervised masked-reconstruction objective (Transformer-MAE, finding 19) does not.
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
N_BASIS = 6


class KANLayer(nn.Module):
    """Simplified KAN edge function: each (in, out) edge is a learnable sum of N_BASIS sine
    harmonics of the input, i.e. phi_{ij}(x) = sum_k a_k sin(k*w*x + b_k). Edges into the same
    output node are summed (the KAN "node = sum of incoming edge functions" rule)."""
    def __init__(self, in_dim, out_dim, n_basis=N_BASIS):
        super().__init__()
        self.in_dim, self.out_dim, self.n_basis = in_dim, out_dim, n_basis
        self.freqs = nn.Parameter(torch.randn(in_dim, out_dim, n_basis) * 1.5)
        self.phases = nn.Parameter(torch.rand(in_dim, out_dim, n_basis) * 2 * np.pi)
        self.amps = nn.Parameter(torch.randn(in_dim, out_dim, n_basis) * (1.0 / np.sqrt(in_dim * n_basis)))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):  # x: (B, in_dim)
        xb = x.unsqueeze(1).unsqueeze(-1)                      # (B, 1, in_dim, 1)
        freqs = self.freqs.permute(1, 0, 2).unsqueeze(0)        # (1, out_dim, in_dim, n_basis)
        phases = self.phases.permute(1, 0, 2).unsqueeze(0)
        amps = self.amps.permute(1, 0, 2).unsqueeze(0)
        xb = x.unsqueeze(1).unsqueeze(-1)                       # (B, 1, in_dim, 1) broadcast vs (1,out,in,basis)
        xb = xb.permute(0, 1, 2, 3)
        activ = amps * torch.sin(freqs * x.unsqueeze(1).unsqueeze(-1) + phases)  # (B, out, in, basis)
        return activ.sum(dim=(2, 3)) + self.bias


class KANScoreNet(nn.Module):
    def __init__(self, in_dim=M, hidden=HIDDEN):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.l1 = KANLayer(in_dim + hidden, hidden)   # <- feature extraction point
        self.l2 = KANLayer(hidden, hidden)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x_t, t_frac, return_feature=False):
        te = self.time_embed(t_frac)
        h1 = self.l1(torch.cat([x_t, te], dim=-1))
        if return_feature:
            return h1
        h2 = self.l2(h1)
        return self.out(h2)


def make_schedule(n=N_TIMESTEPS):
    betas = torch.linspace(1e-4, 0.02, n)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def train_kan(points, epochs=50, lr=3e-3, batch=128, seed=0):
    torch.manual_seed(seed)
    net = KANScoreNet().to(device)
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
def kan_features(net, points, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.tensor(points, dtype=torch.float32).to(device)
    t_frac = torch.full((len(X), 1), 3 / N_TIMESTEPS, device=device)
    noise = torch.randn(X.shape, generator=g, device=device) * np.sqrt(3 / N_TIMESTEPS) * 0.3
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

    net, rows = None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
        if net is None or k % retrain_every == 0:
            net = train_kan(pc[:j+1], epochs=epochs)
        d_euclid = np.sum((pc[:j+1] - pc[j]) ** 2, axis=1)
        idx_euclid = select_analogs(d_euclid, j, last_ok)
        feats = kan_features(net, pc[:j+1])
        d_kan = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_kan = select_analogs(d_kan, j, last_ok)
        if len(idx_euclid) < 3 or len(idx_kan) < 3:
            continue
        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]

        def mean_pred(idx_list):
            rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                    for idx in idx_list if idx + (M-1)*TAU + HORIZON < len(prices)]
            return current_price * np.exp(np.mean(rets)) if rets else None

        euclid_pred, kan_pred = mean_pred(idx_euclid), mean_pred(idx_kan)
        if euclid_pred is None or kan_pred is None:
            continue
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price,
                      "euclid_pred": euclid_pred, "kan_pred": kan_pred})
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
        for col in ("euclid_pred", "kan_pred"):
            up = sub[col] > sub["current_price"]
            rows.append({"combo": combo, "method": col, "dir_acc": float((up == sub["actual_up"]).mean())})
    return pd.DataFrame(rows)


for ticker, name in [("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin")]:
    t0 = time.time()
    df = collect(ticker, name)
    c = cpcv(df)
    e = c[c.method == "euclid_pred"]["dir_acc"]
    kk = c[c.method == "kan_pred"]["dir_acc"]
    print(f"{name:10s} n={len(df):3d}  euclid median={e.median():.3f}  KAN-metric median={kk.median():.3f}  "
          f"KAN loses in {float((kk.to_numpy() < e.to_numpy()).mean()):.2f} of {len(e)} CPCV combos  ({time.time()-t0:.0f}s)")
