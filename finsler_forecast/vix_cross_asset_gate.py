"""
Cross-asset test for S&P 500: every gate/feature tried elsewhere in this directory is derived
from the asset's OWN price series. The literature on S&P 500 short-term prediction repeatedly
points to VIX level/changes as an exogenous, informative cross-asset signal, so this tests VIX
level and 5-day VIX change as gates on karcher, alone and combined with the bootstrap confidence
gate (bootstrap_confidence_gate.py) that already works on this asset.

Result (see README Finding 12): VIX alone is weaker than the existing bootstrap gate (best
bucket 58.3% dir. acc., small n=12), and combining VIX with the bootstrap gate does not beat the
bootstrap gate alone -- the best combined bucket (tight spread + rising VIX) is 57.9%, worse
than the bootstrap gate's standalone 67.6%. VIX doesn't add information on top of the analogs'
own internal agreement for this asset; it mostly just shrinks the sample into noisier buckets.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from data import download_close_prices
from geometry import FinslerRandersGeodesicEngine

rng = np.random.default_rng(0)
vix_sorted = download_close_prices("^VIX", 6)["Close"].sort_index()

def collect(ticker, name, horizon=3, n_scenarios=7, n_folds=80, step_back=2, freq="B", n_boot=200):
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
            engine = (FinslerRandersGeodesicEngine(ticker=ticker, asset_name=name, freq=freq,
                                                     drop_last_bar=False, n_scenarios=n_scenarios)
                      .fetch_and_reconstruct(price_df.iloc[:origin+1]).simulate(horizon))
        except Exception:
            continue
        if engine.k_used != n_scenarios:
            continue
        analog_log_ret = np.log(engine.scenario_prices[:, -1] / current_price)
        boot_spread = float(rng.choice(analog_log_ret, size=(n_boot, n_scenarios), replace=True).mean(axis=1).std())
        d = dates[origin]
        vix_now = vix_sorted.asof(d)
        vix_5d_ago = vix_sorted.asof(d - pd.Timedelta(days=7))
        vix_chg = (vix_now / vix_5d_ago - 1) if vix_5d_ago else np.nan
        rows.append({
            "current_price": current_price, "actual_future_price": actual_future_price,
            "actual_up": actual_future_price > current_price, "karcher_pred": float(engine.karcher_mean_price[-1]),
            "boot_spread": boot_spread, "vix_chg": vix_chg,
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)

def combined_gate(df, min_train=20):
    bs, vc = df["boot_spread"].to_numpy(), df["vix_chg"].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        bs_t, vc_t = np.median(bs[:i]), np.median(vc[:i])
        tight = bs[i] <= bs_t
        rising = vc[i] >= vc_t
        bucket[i] = f"{'tight' if tight else 'wide'}_{'rising' if rising else 'falling'}"
    out = df.copy(); out["bucket"] = bucket
    rows = []
    for b in sorted(set(bucket) - {"warmup"}):
        sub = out[out.bucket == b]
        if sub.empty: continue
        err = sub["karcher_pred"] - sub["actual_future_price"]
        up = sub["karcher_pred"] > sub["current_price"]
        rows.append({"bucket": b, "MAE": err.abs().mean(), "dir_acc": float((up==sub["actual_up"]).mean()), "n": len(sub)})
    return pd.DataFrame(rows)

df = collect("^GSPC", "S&P500")
print(f"n folds: {len(df)}")
print(combined_gate(df).to_string(index=False))
