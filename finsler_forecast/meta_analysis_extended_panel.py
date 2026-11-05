"""
Three follow-ups to the permutation-test finding (statistical_rigor.py):
A) Fisher's method: combine the 5 per-asset p-values into one overall significance test.
B) Extend the asset panel with Silver and WTI crude (2 more assets, independent of Brent/Gold)
   to increase statistical power and test generalization.
C) Investigate the S&P 500 weakness: does the effect size correlate with each asset's return
   autocorrelation (a simple market-efficiency proxy) across the full panel?

Result (see README Finding 32):
A) Fisher's combined test across all 7 assets: chi2=80.06, dof=14, combined p-value ~ 0.000000
   -- decisively significant overall, even though S&P 500 alone wasn't.
B) Silver and WTI crude (2 brand-new assets, never tested before) both show the STRONGEST
   effects of the whole panel (+10.9pp, p<0.0001 and +10.2pp, p<0.0001) -- the effect
   generalizes well; 6 of 7 assets are now individually significant.
C) Correlation between |return autocorrelation| and effect size across all 7 assets: only
   +0.211 -- weak. This simple market-efficiency proxy does NOT explain why S&P 500 is the
   outlier; the real explanation (likely S&P 500's unusually intense arbitrage/informational
   efficiency) remains a hypothesis, not a demonstrated finding.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from statistical_rigor import collect_for_stats, permutation_test
from economic_backtest import download_max

ASSETS = [
    ("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20), ("GC=F", "Gold", 20),
    ("BTC-USD", "Bitcoin", 10), ("ETH-USD", "Ethereum", 8),
    ("SI=F", "Silver", 20), ("CL=F", "WTI Crude", 20),
]

results = {}
for ticker, name, max_years in ASSETS:
    t0 = time.time()
    df = collect_for_stats(ticker, name, max_years=max_years)
    if len(df) < 30:
        print(f"{name}: too few folds, skipping")
        continue
    obs, p, null_dist = permutation_test(df, "atypicality")
    p_floor = max(p, 1.0 / 2001)  # avoid log(0) in Fisher's method
    results[name] = {"gap": obs, "p": p, "p_floor": p_floor, "ticker": ticker}
    print(f"{name:10s} (n={len(df):4d}, {time.time()-t0:.0f}s)  gap={obs*100:+.1f}pp  p={p:.4f}")

print("\n=== A) Fisher's combined test across all assets ===")
p_values = [r["p_floor"] for r in results.values()]
chi2_stat = -2 * sum(np.log(p) for p in p_values)
dof = 2 * len(p_values)
combined_p = 1 - stats.chi2.cdf(chi2_stat, dof)
print(f"  {len(p_values)} assets, chi2={chi2_stat:.2f}, dof={dof}, combined p-value = {combined_p:.6f}")

print("\n=== C) Effect size vs. return autocorrelation (market-efficiency proxy) ===")
for name, r in results.items():
    price_df = download_max(r["ticker"], 6)
    log_ret = np.diff(np.log(price_df["Close"].to_numpy(dtype=np.float64)))
    ac1 = float(np.corrcoef(log_ret[:-1], log_ret[1:])[0, 1])
    print(f"  {name:10s} ac1={ac1:+.4f}  effect_gap={r['gap']*100:+.1f}pp")

gaps = np.array([r["gap"] for r in results.values()])
ac1s = []
for name, r in results.items():
    price_df = download_max(r["ticker"], 6)
    log_ret = np.diff(np.log(price_df["Close"].to_numpy(dtype=np.float64)))
    ac1s.append(float(np.corrcoef(log_ret[:-1], log_ret[1:])[0, 1]))
ac1s = np.array(ac1s)
corr = np.corrcoef(np.abs(ac1s), gaps)[0, 1]
print(f"\n  correlation(|autocorrelation|, effect size) across {len(gaps)} assets = {corr:+.3f}")
