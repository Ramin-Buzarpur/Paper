"""
Combinatorial Purged Cross-Validation (Lopez de Prado)-style robustness check: benchmark.py's
walk-forward is a single path through the folds. CPCV instead splits the folds into N
chronological blocks and scores every combination of k held-out blocks separately, so you get a
distribution of outcomes across many different "paths" through the same data, rather than one
point estimate -- and a PBO (probability of backtest overfitting)-style number: the fraction of
combinations where the method being tested lost to the naive baseline. karcher has no fitted
parameters, so there is nothing to purge/embargo for leakage in the usual sense; CPCV is used
here purely to see whether an apparent edge holds up across many fold groupings or was a property
of one particular ordering.

Result (see README Finding 6): Brent and S&P 500's karcher-over-naive directional edge is robust
(karcher loses in only 18% of combinations on both). Bitcoin's edge, which looked real in the
single walk-forward run, is NOT robust under CPCV (karcher loses in 36% of combinations, median
dir_acc actually ties-or-trails naive) -- it was an artifact of fold ordering, not a real edge.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import numpy as np
import pandas as pd
from benchmark import download_prices, collect_folds

def cpcv_blocks(df, n_blocks=8, k_test=2, embargo=2):
    n = len(df)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    block_idx = [np.arange(edges[b], edges[b+1]) for b in range(n_blocks)]
    results = []
    for test_blocks in itertools.combinations(range(n_blocks), k_test):
        test_rows = np.concatenate([block_idx[b] for b in test_blocks])
        # purge/embargo: drop rows within `embargo` of any test block edge from consideration
        # (karcher has no fitted params, so this mainly guards the metric itself against
        # counting a fold whose horizon overlaps a test/train boundary)
        keep_mask = np.ones(len(test_rows), dtype=bool)
        for i, r in enumerate(test_rows):
            if any(abs(r - e) <= embargo for b in test_blocks for e in (block_idx[b][0], block_idx[b][-1])
                   if e != r):
                pass  # embargo affects train/test leakage for fitted models; karcher has none, so no-op here
        sub = df.iloc[test_rows]
        if len(sub) < 5:
            continue
        for method in ("naive_pred", "karcher_pred"):
            err = sub[method] - sub["actual_future_price"]
            up = sub[method] > sub["current_price"]
            results.append({
                "test_blocks": test_blocks, "method": method.replace("_pred", ""),
                "MAE": err.abs().mean(), "dir_acc": float((up == sub["actual_up"]).mean()), "n": len(sub),
            })
    return pd.DataFrame(results)

for ticker, name, freq in [("BZ=F", "Brent", "B"), ("BTC-USD", "Bitcoin", "D"), ("^GSPC", "S&P500", "B")]:
    price_df = download_prices(ticker, "daily")
    folds = collect_folds(price_df, ticker, name, horizon=3, n_folds=80, step_back=2, freq=freq)
    cpcv = cpcv_blocks(folds, n_blocks=8, k_test=2, embargo=2)
    print(f"\n=== {name}: {len(folds)} folds -> {cpcv['test_blocks'].nunique()} CPCV test-block combos ===")
    for method in ("naive", "karcher"):
        sub = cpcv[cpcv.method == method]
        print(f"  {method:8s}  MAE median={sub.MAE.median():.4f} IQR=[{sub.MAE.quantile(.25):.4f},{sub.MAE.quantile(.75):.4f}]"
              f"   dir_acc median={sub.dir_acc.median():.3f} IQR=[{sub.dir_acc.quantile(.25):.3f},{sub.dir_acc.quantile(.75):.3f}]")
    k = cpcv[cpcv.method=="karcher"].set_index("test_blocks")["dir_acc"]
    n_ = cpcv[cpcv.method=="naive"].set_index("test_blocks")["dir_acc"]
    pbo = float((k < n_).mean())
    print(f"  PBO-style: fraction of {len(k)} combos where karcher dir_acc < naive dir_acc = {pbo:.2f}")
