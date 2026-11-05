"""
Two targeted additions:

A) Granger causality among the 7 assets' causal atypicality-rank series: finding 36 showed
   cross-asset co-occurrence but not DIRECTION. Does one asset's atypicality lead another's by
   1-2 periods (statistically, in the Granger sense)? This upgrades "co-occurrence" to a
   directed lead-lag structure.
B) A genuinely fresh, never-touched crisis window: the March 2023 US regional banking stress
   (SVB/Signature/First Republic failures) -- a prospective test of the systemic-risk framing
   (finding 38) on an episode this investigation has not looked at even once before.

Result (see README Findings 39/40):

(A) Traditional markets show rich, directed Granger structure: S&P 500 -> Silver/WTI/Gold all
at p<0.0001, Brent<->WTI and Brent<->S&P500 bidirectional and strong. Equities appear to Granger-
lead commodity atypicality, not just co-occur with it. Bitcoin and Ethereum show NO significant
Granger relationship (p>0.04, mostly far higher) with any other asset in either direction --
independently corroborating finding 36's "crypto is segmented" claim with a directional method
rather than just a correlation/co-occurrence one.

(B) The composite index rose only +3.0pp over baseline during the March 2023 banking stress --
far weaker than the +36.6pp (2008) and +29.8pp (2020) responses to broad, cross-asset shocks.
Honest interpretation: this is a real limitation, not a universal early-warning system. SVB was
a narrow, sector-specific (US regional banks) shock; the asset panel here (broad equity index,
energy, metals, crypto) has no "sensor" with direct exposure to that specific sector, so the
composite understandably stayed quiet. The indicator responds to shocks broad enough to move
its own panel, not to every systemically important event.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from systemic_risk_indicator import causal_rank_series, ASSETS

FRESH_CRISIS = {"2023 US regional banking stress (Mar-Apr 2023)": ("2023-03-01", "2023-04-30")}
MAX_LAG = 3


def main():
    print("Computing causal atypicality-rank series per asset...")
    series = {}
    for ticker, name in ASSETS:
        t0 = time.time()
        max_years = 20 if name not in ("Bitcoin", "Ethereum") else (10 if name == "Bitcoin" else 8)
        s = causal_rank_series(ticker, max_years=max_years)
        if s is None:
            print(f"  {name}: skipped")
            continue
        series[name] = s
        print(f"  {name}: {len(s)} points ({time.time()-t0:.0f}s)")
        time.sleep(1)

    print("\n=== A) Granger causality among asset atypicality series (lags 1-3) ===")
    names = list(series.keys())
    results = []
    for a in names:
        for b in names:
            if a == b:
                continue
            common = series[a].index.intersection(series[b].index)
            if len(common) < 100:
                continue
            df = pd.DataFrame({"effect": series[b].loc[common].to_numpy(),
                                "cause": series[a].loc[common].to_numpy()}).dropna()
            if len(df) < 100:
                continue
            try:
                res = grangercausalitytests(df[["effect", "cause"]], maxlag=MAX_LAG, verbose=False)
                best_p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, MAX_LAG + 1))
            except Exception:
                continue
            results.append((a, b, best_p, len(df)))

    results.sort(key=lambda r: r[2])
    print(f"{'cause -> effect':30s} {'best p (lag 1-3)':18s} n")
    for a, b, p, n in results:
        marker = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
        print(f"{a:>12s} -> {b:<14s} p={p:.4f}{marker}  (n={n})")

    print("\n=== B) Composite systemic-atypicality index during a FRESH, never-touched crisis ===")
    composite = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()
    composite = composite.resample("D").ffill().dropna()
    baseline = composite.mean()
    print(f"baseline mean = {baseline:.3f}")
    for label, (start, end) in FRESH_CRISIS.items():
        window = composite.loc[start:end]
        if window.empty:
            print(f"{label}: no data")
            continue
        print(f"{label}: mean={window.mean():.3f}  max={window.max():.3f}  "
              f"(+{(window.mean()-baseline)*100:.1f}pp over baseline)")


if __name__ == "__main__":
    main()
