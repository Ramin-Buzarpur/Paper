"""
Does combining the two best gates (volume surprise, finding 14; bootstrap confidence, finding
6) compound into something better than either alone? 2x2 bucket (high/low volume x tight/wide
bootstrap spread).

Result (see README Finding 16): no clear compounding. The "highvol_tight" bucket (both signals
agree it's a good day) is roughly in line with the volume gate alone on S&P 500 (65.2% vs. 60.0%
standalone) and Bitcoin (64.3% vs. 62.5%), not meaingfully better, and on Brent the best bucket
is actually "highvol_wide" (75.0%, n=12) rather than the intersection -- noisy and not a clean
story. Splitting into four buckets mostly just shrinks each one's sample size without a clear
gain; not adopted as an improvement over the volume gate alone.
"""

import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import yfinance as yf
from geometry import FinslerRandersGeodesicEngine

def download_with_volume(ticker, years):
    df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=False)
    df = df[["Close", "Volume"]].dropna(); df = df[df["Close"] > 0].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()

rng = np.random.default_rng(0)

def collect(ticker, name, horizon=3, n_scenarios=7, n_folds=80, step_back=2, n_boot=200):
    df = download_with_volume(ticker, 6)
    prices = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    vol_roll_mean = pd.Series(volume).rolling(20, min_periods=5).mean().to_numpy()
    vol_roll_std = pd.Series(volume).rolling(20, min_periods=5).std().to_numpy()
    vol_z = (volume - vol_roll_mean) / np.where(vol_roll_std > 0, vol_roll_std, 1.0)
    n_total = len(prices); min_history, latest = 200, n_total - horizon - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))
    rows = []
    for origin in origins:
        current_price = prices[origin]; actual = prices[origin + horizon]
        try:
            engine = (FinslerRandersGeodesicEngine(ticker=ticker, asset_name=name, drop_last_bar=False, n_scenarios=n_scenarios)
                      .fetch_and_reconstruct(df[["Close"]].iloc[:origin+1]).simulate(horizon))
        except Exception:
            continue
        if engine.k_used != n_scenarios: continue
        analog_log_ret = np.log(engine.scenario_prices[:, -1] / current_price)
        boot_spread = float(rng.choice(analog_log_ret, size=(n_boot, n_scenarios), replace=True).mean(axis=1).std())
        rows.append({"current_price": current_price, "actual_future_price": actual, "actual_up": actual > current_price,
                      "karcher_pred": float(engine.karcher_mean_price[-1]), "vol_z": vol_z[origin], "boot_spread": boot_spread})
    return pd.DataFrame(rows).dropna().reset_index(drop=True)

def combo_bucket(df, min_train=20):
    vz, bs = df["vol_z"].to_numpy(), df["boot_spread"].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        vz_t, bs_t = np.median(vz[:i]), np.median(bs[:i])
        high_vol = vz[i] >= vz_t; tight = bs[i] <= bs_t
        bucket[i] = f"{'highvol' if high_vol else 'lowvol'}_{'tight' if tight else 'wide'}"
    out = df.copy(); out["bucket"] = bucket
    rows = []
    for b in sorted(set(bucket) - {"warmup"}):
        sub = out[out.bucket == b]
        if sub.empty: continue
        up = sub["karcher_pred"] > sub["current_price"]
        rows.append({"bucket": b, "dir_acc": float((up==sub["actual_up"]).mean()), "n": len(sub)})
    return pd.DataFrame(rows)

for ticker, name in [("^GSPC","S&P500"), ("BZ=F","Brent"), ("BTC-USD","Bitcoin")]:
    df = collect(ticker, name)
    print(f"\n=== {name}: volume-gate x bootstrap-gate combined ===")
    print(combo_bucket(df).to_string(index=False))
