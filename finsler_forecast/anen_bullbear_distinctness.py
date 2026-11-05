"""
Tests whether the diffusion-atypicality signal is just re-discovering classic bull/bear market
regimes (price vs. 200-day moving average, a standard technical regime definition), or whether
it's an independent dimension -- directly relevant to positioning this work as a bridge between
the Analog Ensemble (AnEn) literature (Delle Monache et al. 2011, atmospheric/space-weather
forecasting) and the regime-dependent-predictability literature in finance.

If the atypicality signal is highly correlated with bull/bear state, it adds little beyond
what's already known in finance. If it's roughly independent, AND the predictability gap holds
WITHIN both bull and bear periods separately, that's a stronger, more novel claim.

Result (see README Finding 34): mostly independent, and the effect's regime concentration
differs BY ASSET rather than following one universal pattern -- evidence against "this is just
a relabeled known regime."

| asset | corr(atypicality, bull) | bull-only gap (p) | bear-only gap (p) |
|---|---|---|---|
| Brent | -0.126 | +1.8pp (p=0.322) | **+13.1pp (p<0.0001)** |
| S&P 500 | -0.456 (moderate) | +5.7pp (p=0.019) | -7.1pp (p=0.936, reversed) |
| Bitcoin | +0.008 | **+16.8pp (p<0.0001)** | +6.2pp (p=0.083) |
| Ethereum | +0.033 | +8.0pp (p=0.027) | +7.4pp (p=0.053) |
| Silver | -0.010 | **+14.1pp (p<0.0001)** | +6.9pp (p=0.033) |
| WTI Crude | -0.136 | +6.1pp (p=0.022) | **+12.7pp (p<0.0001)** |

5 of 6 assets show low |correlation| (<0.15) between atypicality and simple trend-following
bull/bear state -- a genuinely different dimension, not a relabeling. The two oil assets
(Brent, WTI) concentrate the effect in bear markets; Bitcoin concentrates it in bull markets;
Silver and Ethereum show it in both. S&P 500 is the exception: moderate correlation (-0.456)
and the effect reverses in bear markets -- consistent with every other finding where S&P 500
behaves differently from the rest of the panel.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from statistical_rigor import collect_for_stats, permutation_test
from economic_backtest import download_max

ASSETS = [
    ("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20),
    ("BTC-USD", "Bitcoin", 10), ("ETH-USD", "Ethereum", 8),
    ("SI=F", "Silver", 20), ("CL=F", "WTI Crude", 20),
]


def collect_with_regime(ticker, name, max_years):
    price_df = download_max(ticker, max_years)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    ma200 = pd.Series(prices).rolling(200, min_periods=200).mean().to_numpy()
    df = collect_for_stats(ticker, name, max_years=max_years)
    # re-derive bull/bear per fold using the SAME non-overlapping origin convention
    # (collect_for_stats doesn't expose origin index, so recompute the fold dates the same way)
    from economic_backtest import embed, M, TAU, HORIZON
    log_ret = np.diff(np.log(prices))
    pc = embed(log_ret)
    n_pts = len(pc)
    min_history, latest = 250, n_pts - HORIZON - 1
    origins = list(range(min_history, latest, HORIZON))
    bull = []
    for j in origins[:len(df)]:
        p = j + (M - 1) * TAU
        bull.append(bool(prices[p] > ma200[p]) if not np.isnan(ma200[p]) else None)
    df = df.iloc[:len(bull)].copy()
    df["bull"] = bull[:len(df)]
    return df.dropna(subset=["bull"]).reset_index(drop=True)


for ticker, name, max_years in ASSETS:
    t0 = time.time()
    df = collect_with_regime(ticker, name, max_years)
    if len(df) < 60:
        continue
    print(f"\n=== {name} (n={len(df)}, {time.time()-t0:.0f}s) ===")

    corr = np.corrcoef(df["atypicality"], df["bull"].astype(float))[0, 1]
    print(f"  correlation(atypicality, bull-regime) = {corr:+.3f}")

    for regime, label in [(True, "bull only"), (False, "bear only")]:
        sub = df[df["bull"] == regime].reset_index(drop=True)
        if len(sub) < 30:
            print(f"  {label:10s}: too few folds ({len(sub)})")
            continue
        obs, p, _ = permutation_test(sub, "atypicality", n_perm=1000)
        print(f"  {label:10s} (n={len(sub):4d}): atypicality gap = {obs*100:+.1f}pp  p={p:.4f}")
