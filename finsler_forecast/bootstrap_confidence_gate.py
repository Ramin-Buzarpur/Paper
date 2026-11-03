"""
Selective-prediction test: instead of correcting karcher's forecast (every attempt at that in
ensemble.py/benchmark.py failed or was a no-op), gate WHEN to trust it, using a confidence
signal that comes from the K analogs' own internal agreement rather than the Finsler wind term
(which already failed as a confidence signal, see README Finding 5).

For each fold, bootstrap-resample the K analog returns with replacement (200 resamples) and take
the std of the resampled means -- a tight spread means the K analogs mostly agree with each
other, a wide spread means they don't. This is available causally (computed only from data up to
the origin date, no look-ahead) and motivated by a real gap noted in the Frechet/Karcher-mean
literature: the empirical mean is a point estimate with no built-in uncertainty quantification,
and bootstrap resampling of the sample is the standard fix.

Result (see README Finding 6): this gate works, substantially, on S&P 500 and Bitcoin (tight
bucket beats wide bucket on directional accuracy by double digits), but is flat-to-reversed on
Brent. Mixed, but for 2 of 3 assets this is the first confidence signal of the many tried here
that actually earns its name.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from data import download_close_prices
from geometry import FinslerRandersGeodesicEngine

def collect(ticker, asset_name, horizon=3, n_scenarios=7, n_folds=80, step_back=2, freq="B", n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    price_df = download_close_prices(ticker, 6)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    dates = price_df.index
    n_total = len(prices)
    min_history = 200
    latest_origin = n_total - horizon - 1
    origins = sorted(sorted(range(latest_origin, min_history, -step_back)[:n_folds]))
    rows = []
    for origin in origins:
        current_price = prices[origin]
        actual_future_price = prices[origin + horizon]
        try:
            engine = (FinslerRandersGeodesicEngine(ticker=ticker, asset_name=asset_name, freq=freq,
                                                     drop_last_bar=False, n_scenarios=n_scenarios)
                      .fetch_and_reconstruct(price_df.iloc[:origin+1]).simulate(horizon))
        except Exception:
            continue
        if engine.k_used != n_scenarios:
            continue
        analog_log_ret = np.log(engine.scenario_prices[:, -1] / current_price)  # (K,)
        boot_means = rng.choice(analog_log_ret, size=(n_boot, n_scenarios), replace=True).mean(axis=1)
        karcher_pred = float(engine.karcher_mean_price[-1])
        rows.append({
            "origin_date": dates[origin], "current_price": current_price,
            "actual_future_price": actual_future_price,
            "actual_up": actual_future_price > current_price,
            "karcher_pred": karcher_pred,
            "boot_spread": float(boot_means.std()),   # causal confidence signal: tight = confident
        })
    return pd.DataFrame(rows)

def gate_breakdown(df, min_train=20):
    spread = df["boot_spread"].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        thresh = np.median(spread[:i])
        bucket[i] = "tight_(confident)" if spread[i] <= thresh else "wide_(unsure)"
    out = df.copy(); out["bucket"] = bucket
    rows = []
    for b in ("tight_(confident)", "wide_(unsure)"):
        sub = out[out.bucket == b]
        if sub.empty: continue
        err = sub["karcher_pred"] - sub["actual_future_price"]
        up = sub["karcher_pred"] > sub["current_price"]
        rows.append({"bucket": b, "MAE": err.abs().mean(), "dir_acc": float((up == sub["actual_up"]).mean()), "n": len(sub)})
    return pd.DataFrame(rows)

for ticker, name, freq in [("BZ=F", "Brent", "B"), ("^GSPC", "S&P500", "B"), ("BTC-USD", "Bitcoin", "D")]:
    df = collect(ticker, name, freq=freq)
    g = gate_breakdown(df)
    print(f"\n=== {name} ({len(df)} folds) — bootstrap-spread confidence gate on karcher ===")
    print(g.to_string(index=False))
