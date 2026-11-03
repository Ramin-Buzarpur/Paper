"""
Gate karcher by realized-volatility regime (recent 21-bar vol, causal expanding terciles) --
a classic, mechanism-different hypothesis from the analog-agreement gates above: calm regimes
should be more analog-forecastable than turbulent ones.

Result (see README Finding 10): no usable signal. The mid/high-vol buckets have too few folds
per asset (as few as n=1) to mean anything, and what numbers exist don't agree in direction
across assets (low-vol best for Brent, mid-vol worst for S&P 500, no differentiation at all for
Bitcoin). Reported as a clean negative rather than cherry-picking the asset where it happens to
look best.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from benchmark import download_prices, collect_folds

def vol_regime_gate(df, min_train=20):
    """Gate on realized volatility regime (recent_vol_21), causally terciled on an expanding
    window -- a completely different mechanism than analog agreement: hypothesis is that
    calm regimes are more analog-forecastable than turbulent ones."""
    vol = df["recent_vol_21"].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        edges = np.quantile(vol[:i], [1/3, 2/3])
        v = vol[i]
        bucket[i] = "low_vol" if v <= edges[0] else ("high_vol" if v > edges[1] else "mid_vol")
    out = df.copy(); out["bucket"] = bucket
    rows = []
    for b in ("low_vol", "mid_vol", "high_vol"):
        sub = out[out.bucket == b]
        if sub.empty: continue
        err = sub["karcher_pred"] - sub["actual_future_price"]
        up = sub["karcher_pred"] > sub["current_price"]
        rows.append({"bucket": b, "MAE": err.abs().mean(), "dir_acc": float((up==sub["actual_up"]).mean()), "n": len(sub)})
    return pd.DataFrame(rows)

for ticker, name, freq in [("BZ=F","Brent","B"), ("^GSPC","S&P500","B"), ("BTC-USD","Bitcoin","D")]:
    price_df = download_prices(ticker, "daily")
    folds = collect_folds(price_df, ticker, name, horizon=3, n_folds=80, step_back=2, freq=freq)
    g = vol_regime_gate(folds)
    print(f"\n=== {name}: realized-volatility-regime gate on karcher ===")
    print(g.to_string(index=False))
