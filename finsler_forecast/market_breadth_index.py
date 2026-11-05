"""
Combines the two scale-up ideas: compute a lightweight atypicality score (diffusion ELBO only,
no classical-detector ensemble, to keep compute feasible) across ~30 diverse S&P 500
constituent stocks, aggregate into a daily "atypicality breadth" (% of stocks flagged atypical
that day -- a novelty-detection analogue of classic advance/decline market-breadth indicators),
and test whether this market-wide breadth signal improves/confirms the S&P 500 INDEX's own
forecast accuracy, beyond the index's own single-asset atypicality signal (finding 31/33).

Result (see README Finding 37): NEGATIVE. Breadth across 30 diverse S&P 500 constituents
(mean=0.477, std=0.411 over 2259 dates) does not meaningfully improve the index's own 3-day
forecast accuracy: gap = +0.2pp, permutation p-value = 0.473 -- indistinguishable from chance.
Reported honestly as a null result, not hidden: aggregating many stocks' idiosyncratic
atypicality does not translate into better index-level directional forecasts, at least with
this lightweight (diffusion-only) per-stock scoring and this specific target (short-horizon
direction accuracy). Contrast with systemic_risk_indicator.py's positive result on a different
target (detecting known crisis periods rather than improving short-horizon accuracy).
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from economic_backtest import (HORIZON, embed, select_analogs, train_scorenet,
                                full_elbo_score, download_max, M, TAU, K, THEILER, N_TIMESTEPS, device)
from statistical_rigor import permutation_test

STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",       # tech/growth
    "JPM", "BAC", "GS", "WFC",                                      # financials
    "XOM", "CVX", "COP",                                            # energy
    "JNJ", "PFE", "UNH", "MRK",                                     # healthcare
    "PG", "KO", "WMT", "COST", "MCD",                               # consumer staples/discretionary
    "CAT", "BA", "GE", "HON",                                       # industrials
    "DIS", "NFLX", "VZ",                                            # media/telecom
]


def download_with_retry(ticker, max_years, tries=4, base_delay=5):
    for attempt in range(tries):
        try:
            d = download_max(ticker, max_years)
            if d is not None and len(d) > 0:
                return d
        except Exception:
            pass
        time.sleep(base_delay * (attempt + 1))
    return None


def lightweight_atypicality(ticker, max_years=15, retrain_every=40, epochs=30):
    df_px = download_with_retry(ticker, max_years)
    if df_px is None:
        return None
    prices = df_px["Close"].to_numpy(dtype=np.float64)
    if len(prices) < 500:
        return None
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    min_history, latest = 250, n_pts - HORIZON - 1
    origins = list(range(min_history, latest, HORIZON))

    net, schedule, rows = None, None, []
    for k, j in enumerate(origins):
        if net is None or k % retrain_every == 0:
            net, schedule = train_scorenet(pc[:j+1], epochs=epochs)
        atyp = full_elbo_score(net, schedule, pc[j], seed=k)
        p = j + (M - 1) * TAU
        rows.append({"date": df_px.index[p], "atyp": atyp})
    return pd.DataFrame(rows)


print("Computing lightweight atypicality for the stock panel...")
series = {}
for tk in STOCKS:
    t0 = time.time()
    s = lightweight_atypicality(tk)
    if s is None:
        print(f"  {tk}: skipped (insufficient data)")
        time.sleep(3)
        continue
    series[tk] = s
    print(f"  {tk}: {len(s)} points ({time.time()-t0:.0f}s)")
    time.sleep(2)  # be gentle with Yahoo Finance's rate limiting

print(f"\nCollected {len(series)} stocks. Building breadth index...")

# causal per-stock flag, then average across stocks per date
flags = {}
for tk, s in series.items():
    vals = s["atyp"].to_numpy()
    flag = np.full(len(vals), np.nan)
    for i in range(20, len(vals)):
        flag[i] = 1.0 if vals[i] >= np.median(vals[:i]) else 0.0
    flags[tk] = pd.Series(flag, index=s["date"]).dropna()

breadth = pd.concat(flags, axis=1).mean(axis=1, skipna=True).dropna()
breadth.name = "breadth"
print(f"Breadth index built on {len(breadth)} dates, mean={breadth.mean():.3f}, std={breadth.std():.3f}")

# S&P 500 index's own forecast + accuracy
spx = download_with_retry("^GSPC", 15)
if spx is None:
    raise SystemExit("could not download ^GSPC after retries")
spx_prices = spx["Close"].to_numpy(dtype=np.float64)
spx_log_ret = np.diff(np.log(spx_prices))
spx_scale = max(float(np.std(spx_log_ret)), 1e-8)
spx_pc = embed(spx_log_ret) / spx_scale
n_pts = len(spx_pc)
min_history, latest = 250, n_pts - HORIZON - 1
origins = list(range(min_history, latest, HORIZON))

net, schedule, rows = None, None, []
for k, j in enumerate(origins):
    if net is None or k % 20 == 0:
        net, schedule = train_scorenet(spx_pc[:j+1], epochs=50)
    last_ok = j - 1
    d = np.sum((spx_pc[:j+1] - spx_pc[j]) ** 2, axis=1)
    idx = select_analogs(d, j, last_ok)
    if len(idx) < 3:
        continue
    p = j + (M - 1) * TAU
    cur, fut = spx_prices[p], spx_prices[p + HORIZON]
    rets = [np.log(spx_prices[ii + (M-1)*TAU + HORIZON] / spx_prices[ii + (M-1)*TAU])
            for ii in idx if ii + (M-1)*TAU + HORIZON < len(spx_prices)]
    if len(rets) < 3:
        continue
    pred_up = np.mean(rets) > 0
    rows.append({"date": spx.index[p], "pred_up": pred_up, "actual_up": fut > cur})

spx_df = pd.DataFrame(rows).set_index("date")
spx_df["correct"] = (spx_df["pred_up"] == spx_df["actual_up"]).astype(int)

merged = spx_df.join(breadth, how="inner").dropna()
print(f"\nMerged S&P500 forecast series with breadth index: {len(merged)} common dates")

obs, p, _ = permutation_test(merged.rename(columns={"breadth": "atypicality"}), "atypicality")
print(f"Breadth-gated S&P500 accuracy gap = {obs*100:+.1f}pp, permutation p-value = {p:.4f}")
