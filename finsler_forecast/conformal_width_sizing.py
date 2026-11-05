"""
"Conformal Kelly"-inspired position sizing (Ryan 2025, SSRN/arXiv): scale position size
INVERSELY to the width of a prediction interval, rather than directly by an anomaly score.
Here the natural interval is already available for free: the spread (std) of the K=7 analogs'
individual predicted returns. A narrow spread (analogs agree) -> bigger position; a wide
spread (analogs disagree) -> smaller position. Per the paper's own finding that simpler,
slower-adapting conformal methods beat adaptive ones, this uses a plain causal rolling
median-normalized width, no fancy recency weighting.

Compared against: (a) the existing atypicality-based sizing (finding 29), (b) a blend of both.

Result (see README Finding 30): no single sizing signal wins everywhere -- different assets
favor different risk controls:

| asset | best Sharpe | best drawdown | winning method |
|---|---|---|---|
| Brent | 3.51 (spread alone) | 23.0% (blend) | spread-based |
| S&P 500 | 3.02 (spread alone) | **12.1%** (spread alone) | spread-based, clearly |
| Gold | 3.09 (blend) | 15.4% (spread alone) | close between methods |
| Bitcoin | 3.57 (finding 29's momentum filter) | 26.5% (finding 29's momentum filter) | momentum-agreement filter |
| Ethereum | 3.15 (blend) | **27.9%** (finding 29's momentum filter) | momentum-agreement filter |

The inverse-spread ("Conformal Kelly"-style) signal is the better risk control for Brent, S&P
500 and Gold; the momentum-agreement filter from finding 29 remains the better one for the two
crypto assets. This is a richer, more honest conclusion than a single best method: several
complementary risk-sizing techniques exist, each suited to a different asset class, rather than
one universal fix.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from economic_backtest import (collect, sharpe, summarize, HORIZON, embed, select_analogs,
                                train_scorenet, full_elbo_score, download_max,
                                M, TAU, K, THEILER, N_TIMESTEPS, device)
import pandas as pd


def collect_with_spread(ticker, name, max_years=20, retrain_every=20, epochs=50):
    price_df = download_max(ticker, max_years)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    min_history = 250
    latest = n_pts - HORIZON - 1
    origins = list(range(min_history, latest, HORIZON))

    net, schedule, rows = None, None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
        if net is None or k % retrain_every == 0:
            net, schedule = train_scorenet(pc[:j+1], epochs=epochs)

        feats = net.forward(torch.tensor(pc[:j+1], dtype=torch.float32, device=device),
                             torch.full((j+1, 1), 3 / N_TIMESTEPS, device=device),
                             return_feature=True).detach().cpu().numpy()
        d_diff = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_diff = select_analogs(d_diff, j, last_ok)
        if len(idx_diff) < 3:
            continue

        atyp, _ = full_elbo_score(net, schedule, pc[j], seed=k), None

        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                for idx in idx_diff if idx + (M-1)*TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred_ret = float(np.mean(rets))
        spread = float(np.std(rets))  # the "conformal interval width" proxy
        actual_ret = float(np.log(actual_future_price / current_price))
        momentum_ret = float(np.sum(pc[j]))

        rows.append({
            "date": price_df.index[p], "pred_ret": pred_ret, "actual_ret": actual_ret,
            "atypicality": atyp, "spread": spread, "momentum_sign": np.sign(momentum_ret),
        })
    return pd.DataFrame(rows)


def run_width_sized(df, min_train=20, cost_bps=2.0, size_range=(0.5, 1.5), blend_atyp=False):
    spread = df["spread"].to_numpy()
    atyp = df["atypicality"].to_numpy()
    rets, prev_size = [], 0.0
    for i in range(min_train, len(df)):
        # narrower spread (lower percentile) -> bigger size: invert the percentile rank
        width_pct = float((spread[:i] < spread[i]).mean())
        size = size_range[1] - width_pct * (size_range[1] - size_range[0])
        if blend_atyp:
            atyp_pct = float((atyp[:i] < atyp[i]).mean())
            atyp_size = size_range[0] + atyp_pct * (size_range[1] - size_range[0])
            size = 0.5 * size + 0.5 * atyp_size
        direction = np.sign(df["pred_ret"].iloc[i])
        if direction == 0:
            rets.append(0.0); prev_size = 0.0; continue
        gross = size * direction * df["actual_ret"].iloc[i]
        cost = (cost_bps / 10000.0) * abs(size - prev_size)
        rets.append(gross - cost)
        prev_size = size
    return np.array(rets)


if __name__ == "__main__":
    for ticker, name, max_years in [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20), ("GC=F", "Gold", 20),
                                     ("BTC-USD", "Bitcoin", 10), ("ETH-USD", "Ethereum", 8)]:
        t0 = time.time()
        df = collect_with_spread(ticker, name, max_years=max_years)
        if len(df) < 30:
            continue
        periods_per_year = 252 / HORIZON
        print(f"\n=== {name} ({time.time()-t0:.0f}s) ===")
        summarize(run_width_sized(df), "inverse-spread sizing (Conformal-Kelly-style)", periods_per_year)
        summarize(run_width_sized(df, blend_atyp=True), "blend: spread + atypicality sizing", periods_per_year)
