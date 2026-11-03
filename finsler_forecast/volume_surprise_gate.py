"""
Volume-surprise gate: a genuinely different kind of signal from everything else tried in this
directory -- trading VOLUME, not a transform of price/returns. Feature: today's volume as a
z-score against its trailing 20-day mean/std (causal). Gate karcher on a causal expanding-
median split of that z-score (high vs. low volume surprise).

Result (see README Finding 14) -- the strongest, most universal result in this whole
investigation: unlike every other gate tried (wind t-stat, bootstrap spread, raw analog spread,
realized-vol regime, VIX), this one is CPCV-robust on ALL THREE assets tested, not just one or
two:

| asset | high-volume-surprise median dir. acc. (CPCV) | low-volume-surprise median |
|---|---|---|
| S&P 500 | 60.0% | 37.5% |
| Brent | 70.7% | 50.0% |
| Bitcoin | 62.5% | 44.4% |

Mechanism plausibility: unusually high volume is a classic proxy for new information hitting
the market (earnings, news, institutional flow) -- days like that are exactly when a short-term
analog/momentum continuation is more likely to actually play out, vs. a quiet low-volume day
where the next move is closer to noise. This single preprocessing-free gate beats every
model-based correction tried in findings 1-13.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import yfinance as yf

def download_with_volume(ticker, years):
    df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=False)
    df = df[["Close", "Volume"]].dropna()
    df = df[df["Close"] > 0].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()

def collect(ticker, name, horizon=3, n_folds=80, step_back=2):
    df = download_with_volume(ticker, 6)
    prices = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    # volume z-score vs trailing 20-day average (causal, at each day)
    vol_roll_mean = pd.Series(volume).rolling(20, min_periods=5).mean().to_numpy()
    vol_roll_std = pd.Series(volume).rolling(20, min_periods=5).std().to_numpy()
    vol_z = (volume - vol_roll_mean) / np.where(vol_roll_std > 0, vol_roll_std, 1.0)

    n_total = len(prices); min_history, latest = 200, n_total - horizon - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))
    rows = []
    # reuse geometry engine's karcher for the forecast itself
    from geometry import FinslerRandersGeodesicEngine
    for origin in origins:
        current_price = prices[origin]; actual = prices[origin + horizon]
        try:
            engine = (FinslerRandersGeodesicEngine(ticker=ticker, asset_name=name, drop_last_bar=False)
                      .fetch_and_reconstruct(df[["Close"]].iloc[:origin+1]).simulate(horizon))
        except Exception:
            continue
        rows.append({
            "current_price": current_price, "actual_future_price": actual, "actual_up": actual > current_price,
            "karcher_pred": float(engine.karcher_mean_price[-1]), "vol_z": vol_z[origin],
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)

def gate(df, col, min_train=20):
    vals = df[col].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        thresh = np.median(vals[:i])
        bucket[i] = "high" if vals[i] >= thresh else "low"
    out = df.copy(); out["bucket"] = bucket
    rows = []
    for b in ("high", "low"):
        sub = out[out.bucket == b]
        if sub.empty: continue
        err = sub["karcher_pred"] - sub["actual_future_price"]
        up = sub["karcher_pred"] > sub["current_price"]
        rows.append({"bucket": b, "MAE": err.abs().mean(), "dir_acc": float((up==sub["actual_up"]).mean()), "n": len(sub)})
    return pd.DataFrame(rows)

for ticker, name in [("^GSPC","S&P500"), ("BZ=F","Brent"), ("BTC-USD","Bitcoin")]:
    df = collect(ticker, name)
    print(f"\n=== {name}: volume-surprise (z-score vs 20d avg) gate on karcher ===")
    print(gate(df, "vol_z").to_string(index=False))

print("\n\n=== CPCV robustness check on the volume-surprise gate ===")
import itertools
def cpcv_bucketed(df, n_blocks=8, k_test=2, min_train=20):
    vals = df["vol_z"].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        thresh = np.median(vals[:i])
        bucket[i] = "high" if vals[i] >= thresh else "low"
    work = df.iloc[min_train:].copy()
    work["bucket"] = bucket[min_train:]
    n = len(work); edges = np.linspace(0, n, n_blocks+1).astype(int)
    blocks = [np.arange(edges[b], edges[b+1]) for b in range(n_blocks)]
    rows = []
    for combo in itertools.combinations(range(n_blocks), k_test):
        idx = np.concatenate([blocks[b] for b in combo])
        sub = work.iloc[idx]
        for b in ("high", "low"):
            bsub = sub[sub.bucket == b]
            if len(bsub) < 4: continue
            up = bsub["karcher_pred"] > bsub["current_price"]
            rows.append({"combo": combo, "bucket": b, "dir_acc": float((up == bsub["actual_up"]).mean()), "n": len(bsub)})
    return pd.DataFrame(rows)

for ticker, name in [("^GSPC","S&P500"), ("BZ=F","Brent"), ("BTC-USD","Bitcoin")]:
    df = collect(ticker, name)
    c = cpcv_bucketed(df)
    h = c[c.bucket=="high"]["dir_acc"]; l = c[c.bucket=="low"]["dir_acc"]
    print(f"{name:10s} high median={h.median():.3f} (n_combos={len(h)})  low median={l.median():.3f} (n_combos={len(l)})")
