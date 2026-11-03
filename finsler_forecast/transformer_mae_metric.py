"""
Transformer self-supervised embedding as an analog-similarity metric -- parallel idea to
diffusion_score_metric.py, swapping the representation learner: instead of a diffusion
denoiser, a small Transformer encoder is pretrained with a masked-autoencoding pretext task
(randomly mask one of the 3 delay-embedding coordinates, predict it from the other two) on the
engine's own point cloud. The encoder's output embedding becomes the similarity feature for
picking the K=7 analogs; their true historical continuations are averaged exactly as karcher
does. Tests whether a Transformer-learned representation helps where diffusion's did (S&P 500)
or elsewhere.
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
D_MODEL = 32


class MAEEncoder(nn.Module):
    """Treats the 3 delay coordinates as a length-3 'sequence', each coordinate projected to
    d_model, self-attention across them, reconstructs the masked coordinate."""
    def __init__(self, d_model=D_MODEL, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, M, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64,
                                                dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.recon_head = nn.Linear(d_model, 1)

    def forward(self, x, mask_idx=None):  # x: (B, 3)
        h = self.proj(x.unsqueeze(-1)) + self.pos  # (B, 3, d_model)
        if mask_idx is not None:
            b = torch.arange(len(x), device=x.device)
            h = h.clone()
            h[b, mask_idx] = self.mask_token.expand(len(x), 1, -1).squeeze(1)
        z = self.encoder(h)
        return z

    def embedding(self, x):
        z = self.forward(x, mask_idx=None)
        return z.mean(dim=1)  # (B, d_model) -- pooled representation, the similarity feature

    def reconstruct(self, x, mask_idx):
        z = self.forward(x, mask_idx=mask_idx)
        b = torch.arange(len(x), device=x.device)
        return self.recon_head(z[b, mask_idx]).squeeze(-1)


def train_mae(points, epochs=50, lr=2e-3, batch=128, seed=0):
    torch.manual_seed(seed)
    net = MAEEncoder().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    X = torch.tensor(points, dtype=torch.float32).to(device)
    n = len(X)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x = X[idx]
            mask_idx = torch.randint(0, M, (len(idx),), device=device)
            target = x[torch.arange(len(idx), device=device), mask_idx]
            pred = net.reconstruct(x, mask_idx)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def mae_features(net, points):
    X = torch.tensor(points, dtype=torch.float32).to(device)
    return net.embedding(X).cpu().numpy()


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
            net = train_mae(pc[:j+1], epochs=epochs)

        d_euclid = np.sum((pc[:j+1] - pc[j]) ** 2, axis=1)
        idx_euclid = select_analogs(d_euclid, j, last_ok)
        feats = mae_features(net, pc[:j+1])
        d_mae = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_mae = select_analogs(d_mae, j, last_ok)
        if len(idx_euclid) < 3 or len(idx_mae) < 3:
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

        euclid_pred, mae_pred = mean_pred(idx_euclid), mean_pred(idx_mae)
        if euclid_pred is None or mae_pred is None:
            continue
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price,
                      "euclid_pred": euclid_pred, "mae_pred": mae_pred})
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
        for col in ("euclid_pred", "mae_pred"):
            up = sub[col] > sub["current_price"]
            rows.append({"combo": combo, "method": col, "dir_acc": float((up == sub["actual_up"]).mean())})
    return pd.DataFrame(rows)


for ticker, name in [("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin")]:
    t0 = time.time()
    df = collect(ticker, name)
    c = cpcv(df)
    e = c[c.method == "euclid_pred"]["dir_acc"]
    m = c[c.method == "mae_pred"]["dir_acc"]
    print(f"{name:10s} n={len(df):3d}  euclid median={e.median():.3f}  transformer-MAE-metric median={m.median():.3f}  "
          f"MAE-metric loses in {float((m.to_numpy() < e.to_numpy()).mean()):.2f} of {len(e)} CPCV combos  ({time.time()-t0:.0f}s)")
