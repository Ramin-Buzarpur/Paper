"""
VAE-based similarity metric for analog selection -- third representation learner tried as a
metric (after diffusion in diffusion_score_metric.py [worked, S&P500] and Transformer MAE in
transformer_mae_metric.py [failed everywhere]). A VAE encodes the 3D embedding into a latent
mean vector; that latent mean becomes the similarity feature for picking K=7 analogs, exactly
as before -- true historical continuations averaged, only the similarity space changes.

Result (see README Finding 22), CPCV-confirmed: unlike diffusion (strong, asset-specific win)
and Transformer-MAE (loses everywhere), VAE is a mild, broadly SAFE improvement -- never loses
badly anywhere, never wins big anywhere:

| asset | plain-Euclidean | VAE-metric | VAE loses in |
|---|---|---|---|
| Brent | 75.0% | 72.5% | 25% of combos |
| S&P 500 | 57.5% | 60.0% | 0% of combos |
| Bitcoin | 67.5% | 65.0% | 25% of combos |
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
LATENT = 8


class VAE(nn.Module):
    def __init__(self, in_dim=M, hidden=32, latent=LATENT):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, in_dim))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, logvar


def train_vae(points, epochs=60, lr=2e-3, batch=128, beta=0.1, seed=0):
    torch.manual_seed(seed)
    net = VAE().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    X = torch.tensor(points, dtype=torch.float32).to(device)
    n = len(X)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x = X[idx]
            recon, mu, logvar = net(x)
            recon_loss = nn.functional.mse_loss(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def vae_features(net, points):
    X = torch.tensor(points, dtype=torch.float32).to(device)
    mu, _ = net.encode(X)
    return mu.cpu().numpy()


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


def collect(ticker, name, n_folds=80, step_back=2, retrain_every=15, epochs=60):
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
            net = train_vae(pc[:j+1], epochs=epochs)
        d_euclid = np.sum((pc[:j+1] - pc[j]) ** 2, axis=1)
        idx_euclid = select_analogs(d_euclid, j, last_ok)
        feats = vae_features(net, pc[:j+1])
        d_vae = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_vae = select_analogs(d_vae, j, last_ok)
        if len(idx_euclid) < 3 or len(idx_vae) < 3:
            continue
        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]

        def mean_pred(idx_list):
            rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                    for idx in idx_list if idx + (M-1)*TAU + HORIZON < len(prices)]
            return current_price * np.exp(np.mean(rets)) if rets else None

        euclid_pred, vae_pred = mean_pred(idx_euclid), mean_pred(idx_vae)
        if euclid_pred is None or vae_pred is None:
            continue
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price,
                      "euclid_pred": euclid_pred, "vae_pred": vae_pred})
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
        for col in ("euclid_pred", "vae_pred"):
            up = sub[col] > sub["current_price"]
            rows.append({"combo": combo, "method": col, "dir_acc": float((up == sub["actual_up"]).mean())})
    return pd.DataFrame(rows)


for ticker, name in [("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin")]:
    t0 = time.time()
    df = collect(ticker, name)
    c = cpcv(df)
    e = c[c.method == "euclid_pred"]["dir_acc"]
    v = c[c.method == "vae_pred"]["dir_acc"]
    print(f"{name:10s} n={len(df):3d}  euclid median={e.median():.3f}  VAE-metric median={v.median():.3f}  "
          f"VAE loses in {float((v.to_numpy() < e.to_numpy()).mean()):.2f} of {len(e)} CPCV combos  ({time.time()-t0:.0f}s)")
