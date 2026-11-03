"""
Split-conformal-style selective gate, using the raw spread of the K analogs' own horizon
returns (not the bootstrap-resampled spread of their MEAN, which is bootstrap_confidence_gate.py
and already works on 2/3 assets) as the nonconformity proxy, calibrated on an expanding window
strictly before the test fold.

Result (see README Finding 9): weaker and less consistent than the bootstrap gate. Reversed on
Brent (narrow/confident bucket is WORSE: 42.9% vs 75.0% dir. acc. in the wide bucket), flat on
S&P 500, and directionally consistent with the bootstrap gate's result on Bitcoin (60.0% vs
46.7%) but smaller. Raw analog spread is evidently a noisier confidence proxy than the
bootstrap-resampled spread of the mean -- bootstrapping is doing real work here, not just
relabeling the same signal.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from benchmark import download_prices, collect_folds

def conformal_gate(df, min_train=20, cal_frac=0.5):
    """Split-conformal: calibrate |karcher_err| quantile on an expanding calibration window
    (strictly before the test fold), then bucket the test fold by whether its own |karcher_err|
    prediction proxy (here, analog_spread -- the only causally available proxy for the error
    magnitude) falls below the calibrated median of analog_spread itself. This differs from the
    bootstrap gate: the threshold is picked to target a nonconformity QUANTILE (e.g. the level
    that would have bounded 50% of past errors), not just a median split of the signal."""
    rows = []
    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i]
        cal = train.iloc[-max(10, int(len(train)*cal_frac)):]
        thresh = np.quantile(cal["analog_spread"].to_numpy(), 0.5)
        bucket = "narrow" if test["analog_spread"] <= thresh else "wide"
        rows.append({**test.to_dict(), "bucket": bucket})
    out = pd.DataFrame(rows)
    res = []
    for b in ("narrow", "wide"):
        sub = out[out.bucket == b]
        if sub.empty: continue
        err = sub["karcher_pred"] - sub["actual_future_price"]
        up = sub["karcher_pred"] > sub["current_price"]
        res.append({"bucket": b, "MAE": err.abs().mean(), "dir_acc": float((up==sub["actual_up"]).mean()), "n": len(sub)})
    return pd.DataFrame(res)

for ticker, name, freq in [("BZ=F","Brent","B"), ("^GSPC","S&P500","B"), ("BTC-USD","Bitcoin","D")]:
    price_df = download_prices(ticker, "daily")
    folds = collect_folds(price_df, ticker, name, horizon=3, n_folds=80, step_back=2, freq=freq)
    g = conformal_gate(folds)
    print(f"\n=== {name}: conformal-style analog_spread gate ===")
    print(g.to_string(index=False))
