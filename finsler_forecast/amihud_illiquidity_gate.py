"""
Amihud illiquidity (|return| / dollar volume, a standard asset-pricing liquidity measure) as a
gate -- a different functional combination of price and volume than the plain volume-surprise
gate (volume_surprise_gate.py, finding 14).

Result (see README Finding 15): weaker than and inconsistent with the volume gate. The sign
flips across assets -- low illiquidity (calmer, more liquid days) is better for S&P 500 (57.1%
vs. 46.9%) and Brent (66.7% vs. 50.0%), but HIGH illiquidity is better for Bitcoin (61.5% vs.
44.1%). A gate whose useful direction isn't even consistent in sign across assets isn't a
usable universal signal the way the volume-surprise gate is.
"""

import sys, os
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

def collect(ticker, name, horizon=3, n_folds=80, step_back=2):
    df = download_with_volume(ticker, 6)
    prices = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    illiq = np.abs(log_ret) / np.where(volume[1:] > 0, volume[1:], 1.0) * 1e9
    illiq_s = pd.Series(illiq)
    illiq_mean = illiq_s.rolling(20, min_periods=5).mean().to_numpy()
    illiq_std = illiq_s.rolling(20, min_periods=5).std().to_numpy()
    illiq_z = (illiq - illiq_mean) / np.where(illiq_std > 0, illiq_std, 1.0)
    n_total = len(prices); min_history, latest = 200, n_total - horizon - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))
    rows = []
    for origin in origins:
        current_price = prices[origin]; actual = prices[origin + horizon]
        try:
            engine = (FinslerRandersGeodesicEngine(ticker=ticker, asset_name=name, drop_last_bar=False)
                      .fetch_and_reconstruct(df[["Close"]].iloc[:origin+1]).simulate(horizon))
        except Exception:
            continue
        idx_ret = origin - 1
        if idx_ret < 0 or idx_ret >= len(illiq_z) or not np.isfinite(illiq_z[idx_ret]): continue
        rows.append({"current_price": current_price, "actual_future_price": actual, "actual_up": actual > current_price,
                      "karcher_pred": float(engine.karcher_mean_price[-1]), "illiq_z": illiq_z[idx_ret]})
    return pd.DataFrame(rows)

def gate(df, col, min_train=20):
    vals = df[col].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        thresh = np.median(vals[:i]); bucket[i] = "high" if vals[i] >= thresh else "low"
    out = df.copy(); out["bucket"] = bucket
    rows = []
    for b in ("high", "low"):
        sub = out[out.bucket == b]
        if sub.empty: continue
        up = sub["karcher_pred"] > sub["current_price"]
        rows.append({"bucket": b, "dir_acc": float((up==sub["actual_up"]).mean()), "n": len(sub)})
    return pd.DataFrame(rows)

for ticker, name in [("^GSPC","S&P500"), ("BZ=F","Brent"), ("BTC-USD","Bitcoin")]:
    df = collect(ticker, name)
    print(f"\n=== {name}: Amihud illiquidity z-score gate ===")
    print(gate(df, "illiq_z").to_string(index=False))
