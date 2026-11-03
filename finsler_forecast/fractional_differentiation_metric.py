"""
Fractional differentiation (Lopez de Prado, "Advances in Financial Machine Learning" ch. 5) as
the input representation for analog selection, instead of plain log-returns (d=1 differencing,
which the fracdiff literature argues over-differences and destroys memory/predictive signal).

Mechanism: Fixed-Width Window fracdiff (FFD) -- weights w_k = (-1)^k * C(d,k), truncated at a
weight-loss threshold, convolved with log-prices. At each origin, d* is chosen CAUSALLY (grid
search over d in [0.1..1.0] on the expanding history up to that origin only, ADF test, pick the
smallest d with p < 0.05 -- the standard de Prado selection rule: minimum differencing for
stationarity, maximum memory retained). The resulting fractionally-differenced series replaces
log-returns as the embedding input, both for a plain-Euclidean (no network) metric and for the
diffusion-metric-learner (with network, same mechanism as finding 18).

Caught two bugs during development, both worth noting: an index-alignment mismatch between the
return-space and fracdiff-space embeddings initially dropped every origin index out of range
(the fracdiff array was only extended on the (infrequent) days d* was re-picked, so the ~14
folds between re-picks ran against a too-short array and silently returned 0 analogs each time
-- caught because the result set was almost entirely empty). Fixed by decoupling the expensive
d* search (periodic) from the cheap fracdiff-series computation itself (every fold, always long
enough to cover the current origin).

Result (see README Finding 25), CPCV-confirmed, causal d* (typically landing around 0.2-0.3,
confirming de Prado's finding that d=1 over-differences): no improvement over what's already
known.

| asset | plain log-return | fracdiff (no network) | fracdiff (with network) |
|---|---|---|---|
| Brent | 75.0% | 65.0% (loses 75%) | 70.0% (loses 46%) |
| S&P 500 | 57.5% | 60.0% (loses 43%) | 65.0% (loses 0%, but ties finding 18's level, doesn't beat it) |
| Bitcoin | 67.5% | 62.5% (loses 64%) | 65.0% (loses 64%) |

The de Prado memory-preservation argument is well-established in the literature, but on this
specific task (3-day-ahead analog similarity), it doesn't add information beyond what plain
log-returns plus a diffusion-learned metric already capture.
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import adfuller
from data import download_close_prices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M, TAU, K, THEILER, HORIZON = 3, 1, 7, 10, 3
N_TIMESTEPS = 50
HIDDEN = 32


def fracdiff_weights(d, thresh=1e-4, max_window=100):
    w = [1.0]
    k = 1
    while k < max_window:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w)


def fracdiff_series(log_prices, d, thresh=1e-4):
    w = fracdiff_weights(d, thresh)
    width = len(w)
    n = len(log_prices)
    out = np.full(n, np.nan)
    for t in range(width - 1, n):
        out[t] = np.dot(w, log_prices[t - width + 1:t + 1][::-1])
    return out


def pick_d_star(log_prices_causal, d_grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0), p_thresh=0.05):
    for d in d_grid:
        series = fracdiff_series(log_prices_causal, d)
        series = series[~np.isnan(series)]
        if len(series) < 50:
            continue
        try:
            pval = adfuller(series, maxlag=1, autolag=None)[1]
        except Exception:
            continue
        if pval < p_thresh:
            return d
    return 1.0  # fall back to full differencing if nothing else reaches stationarity


def embed(series):
    n = len(series)
    n_pts = n - (M - 1) * TAU
    return np.column_stack([series[(M-1-i)*TAU : (M-1-i)*TAU+n_pts] for i in range(M)])


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
    return net


@torch.no_grad()
def net_features(net, points, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.tensor(points, dtype=torch.float32).to(device)
    t_frac = torch.full((len(X), 1), 3 / N_TIMESTEPS, device=device)
    noise = torch.randn(X.shape, generator=g, device=device) * np.sqrt(3 / N_TIMESTEPS) * 0.3
    return net(X + noise, t_frac, return_feature=True).cpu().numpy()


def collect(ticker, name, n_folds=80, step_back=2, retrain_every=15, epochs=50, d_recompute_every=20):
    price_df = download_close_prices(ticker, 6)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_prices = np.log(prices)
    log_ret = np.diff(log_prices)
    scale = max(float(np.std(log_ret)), 1e-8)
    pc_ret = embed(log_ret) / scale  # baseline: plain log-return embedding

    n_pts_ret = len(pc_ret)
    min_history, latest = 200, n_pts_ret - HORIZON - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))

    net, d_star, frac_pc_full = None, None, None
    rows = []
    for k, j in enumerate(origins):
        last_ok = j - 1
        price_idx = j + (M - 1) * TAU + 1  # index into `prices` corresponding to origin "today"

        # d* (the expensive ADF-based search) updates only periodically; the fracdiff series
        # ITSELF is recomputed every fold up to the current price_idx, using the current d*,
        # so its length always covers the current origin j (fixed bug: previously the series
        # was only extended when d* was re-picked, so every fold between re-picks had a too-
        # short array and was silently skipped -- caught because almost every fold was empty).
        if d_star is None or k % d_recompute_every == 0:
            d_star = pick_d_star(log_prices[: price_idx + 1])
        fd = fracdiff_series(log_prices[: price_idx + 1], d_star)
        # drop index 0 so fd[1:] aligns 1:1 with log_ret (same convention as pc_ret, same j)
        fd_ret = fd[1:]
        frac_scale = max(float(np.nanstd(fd_ret[~np.isnan(fd_ret)])), 1e-8)
        frac_pc_full = embed(fd_ret) / frac_scale  # same index j as pc_ret by construction

        if net is None or k % retrain_every == 0:
            net = train_scorenet(pc_ret[:j+1], epochs=epochs)

        if j >= len(frac_pc_full) or np.isnan(frac_pc_full[j]).any():
            continue
        valid_mask = ~np.isnan(frac_pc_full[:j+1]).any(axis=1)

        d_euclid = np.sum((pc_ret[:j+1] - pc_ret[j]) ** 2, axis=1)
        idx_euclid = select_analogs(d_euclid, j, last_ok)

        d_frac = np.where(valid_mask, np.sum((frac_pc_full[:j+1] - frac_pc_full[j]) ** 2, axis=1), np.inf)
        idx_frac = select_analogs(d_frac, j, last_ok)

        feats = net_features(net, pc_ret[:j+1])
        d_net = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_net = select_analogs(d_net, j, last_ok)

        if min(len(idx_euclid), len(idx_frac), len(idx_net)) < 3:
            continue

        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]

        def mean_pred(idx_list):
            rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                    for idx in idx_list if idx + (M-1)*TAU + HORIZON < len(prices)]
            return current_price * np.exp(np.mean(rets)) if rets else None

        euclid_pred, frac_pred, net_pred = mean_pred(idx_euclid), mean_pred(idx_frac), mean_pred(idx_net)
        if None in (euclid_pred, frac_pred, net_pred):
            continue
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price,
                      "euclid_pred": euclid_pred, "frac_pred": frac_pred, "net_pred": net_pred,
                      "d_star": d_star})
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
        for col in ("euclid_pred", "frac_pred", "net_pred"):
            up = sub[col] > sub["current_price"]
            rows.append({"combo": combo, "method": col, "dir_acc": float((up == sub["actual_up"]).mean())})
    return pd.DataFrame(rows)


for ticker, name in ([("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin")] if __name__ == "__main__" else []):
    t0 = time.time()
    df = collect(ticker, name)
    print(f"{name}: df rows = {len(df)}")
    if len(df) < 10:
        print(f"{name}: too few usable folds ({len(df)})")
        continue
    c = cpcv(df)
    e = c[c.method == "euclid_pred"]["dir_acc"].to_numpy()
    f = c[c.method == "frac_pred"]["dir_acc"].to_numpy()
    nn_ = c[c.method == "net_pred"]["dir_acc"].to_numpy()
    print(f"{name:10s} n={len(df):3d}  d*~{df['d_star'].mean():.2f}  "
          f"plain-return={np.median(e):.3f}  fracdiff(no-net)={np.median(f):.3f} (loses {float((f<e).mean()):.2f})  "
          f"fracdiff(net)={np.median(nn_):.3f} (loses {float((nn_<e).mean()):.2f})  ({time.time()-t0:.0f}s)")
