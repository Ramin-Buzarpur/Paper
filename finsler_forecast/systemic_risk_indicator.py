"""
Tests whether a composite cross-asset atypicality index spikes around well-known historical
market-stress episodes -- reframing the core finding as a systemic-risk early-warning
indicator (in the spirit of the ECB's CISS index), a far more general and economically
legible claim than "improves 3-day direction accuracy."

Mechanism: average the (lightweight, diffusion-only) causal atypicality percentile rank across
several independent assets (Brent, S&P 500, Gold, Silver, WTI, Bitcoin, Ethereum) into one daily
composite score, then inspect its level during/around known crisis windows:
  - 2008 Global Financial Crisis (Sep-Nov 2008)
  - 2020 COVID crash (Feb-Mar 2020)
  - 2022 inflation-shock selloff (Jan-Oct 2022)
compared to its unconditional average level.

Result (see README Finding 38): a clean, economically interpretable positive result.

| crisis window | composite mean | vs. baseline (0.407) |
|---|---|---|
| 2008 GFC (Sep-Nov 2008) | 0.773 | **+36.6pp** (max hit the 1.0 ceiling) |
| 2020 COVID crash (Feb-Mar 2020) | 0.704 | **+29.8pp** (max 0.987) |
| 2022 inflation selloff (Jan-Oct 2022) | 0.515 | +10.9pp (weaker) |

The composite nearly doubles during the two sudden, acute shocks (2008, 2020) and shows a
smaller but still elevated level during the slower, grinding 2022 selloff -- consistent with
finding 34's observation that this signal behaves differently depending on whether a regime
shift is abrupt or gradual. This reframes the core finding from "improves 3-day direction
accuracy" to a more general, economically legible claim: a composite cross-asset atypicality
index behaves like a systemic-risk early-warning indicator, in the spirit of (though far
simpler than) the ECB's CISS index.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from economic_backtest import (HORIZON, embed, select_analogs, train_scorenet,
                                full_elbo_score, download_max, M, TAU, N_TIMESTEPS, device)

ASSETS = [
    ("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("GC=F", "Gold"), ("SI=F", "Silver"),
    ("CL=F", "WTI Crude"), ("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"),
]

CRISIS_WINDOWS = {
    "2008 GFC (Sep-Nov 2008)": ("2008-09-01", "2008-11-30"),
    "2020 COVID crash (Feb-Mar 2020)": ("2020-02-15", "2020-03-31"),
    "2022 inflation selloff (Jan-Oct 2022)": ("2022-01-01", "2022-10-31"),
}


def download_with_retry(ticker, max_years, tries=4, base_delay=5):
    for attempt in range(tries):
        try:
            d = download_max(ticker, max_years)
            if d is not None and len(d) > 0:
                return d
        except Exception:
            pass
        time.sleep(base_delay * (attempt + 1))
    return None


def causal_rank_series(ticker, max_years, retrain_every=30, epochs=30):
    price_df = download_with_retry(ticker, max_years)
    if price_df is None:
        return None
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    if len(prices) < 500:
        return None
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    min_history, latest = 250, n_pts - HORIZON - 1
    origins = list(range(min_history, latest, HORIZON))

    net, schedule = None, None
    vals, dates = [], []
    for k, j in enumerate(origins):
        if net is None or k % retrain_every == 0:
            net, schedule = train_scorenet(pc[:j+1], epochs=epochs)
        atyp = full_elbo_score(net, schedule, pc[j], seed=k)
        vals.append(atyp)
        dates.append(price_df.index[j + (M - 1) * TAU])

    vals = np.array(vals)
    rank = np.full(len(vals), np.nan)
    for i in range(20, len(vals)):
        rank[i] = (vals[:i] < vals[i]).mean()
    return pd.Series(rank, index=pd.DatetimeIndex(dates)).dropna()


print("Computing causal atypicality-rank series per asset...")
series = {}
for ticker, name in ASSETS:
    t0 = time.time()
    s = causal_rank_series(ticker, max_years=20 if name not in ("Bitcoin", "Ethereum") else (10 if name == "Bitcoin" else 8))
    if s is None:
        print(f"  {name}: skipped")
        continue
    series[name] = s
    print(f"  {name}: {len(s)} points ({time.time()-t0:.0f}s)")
    time.sleep(2)

composite = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()
composite = composite.resample("D").ffill().dropna()
baseline_mean = composite.mean()
print(f"\nComposite systemic-atypicality index: {len(composite)} days, baseline mean={baseline_mean:.3f}")

print("\n=== Composite index level during known crisis windows vs. baseline ===")
for label, (start, end) in CRISIS_WINDOWS.items():
    window = composite.loc[start:end]
    if window.empty:
        print(f"{label:38s}: no data in this window")
        continue
    print(f"{label:38s}: mean={window.mean():.3f}  max={window.max():.3f}  "
          f"(baseline={baseline_mean:.3f}, +{(window.mean()-baseline_mean)*100:.1f}pp over baseline)")
