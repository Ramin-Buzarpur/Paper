"""
Prompted by finding 49's result on the LLE mechanism: rather than hunting for a replacement
chaos/complexity metric (a real risk of turning into exactly the kind of multiple-method
fishing expedition this investigation has spent findings 45/49/50 trying to avoid), apply the
SAME surrogate-data methodology directly to the paper's actual central claim (finding 33: the
atypicality-conditioned forecast-accuracy gap) instead of to an auxiliary explanatory feature.

This is a meaningfully different test than finding 49's. The LLE gap (finding 49) is a purely
geometric, self-referential statistic of the embedded point cloud -- it never touches a real
future outcome. The core claim's accuracy gap DOES touch a real, external outcome: whether the
analog forecast's predicted direction actually matched the asset's real future return. A
linear AAFT surrogate, by construction, has no serial dependence beyond its (matched) linear
autocorrelation -- so if the real accuracy gap is driven by genuine higher-order/nonlinear
predictability structure (as opposed to something a linear process could produce), it should
look different from the surrogate null, unlike the self-referential LLE statistic did.

Method: for each of 2000 permutation-style evaluations, generate an AAFT surrogate return
series (same procedure as finding 49), run the actual causal analog-forecasting pipeline
(embed, select causal k-NN analogs, predict direction, score vs. actual direction) and the
same cheap causal k-NN atypicality score used in finding 49 (for tractability), then compute
the atypicality-conditioned accuracy gap exactly as in finding 33/statistical_rigor.py. Compare
the real gap's rank against the surrogate null.

Result (see README Finding 51): printed output / README has the actual numbers.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from economic_backtest import embed, download_max, select_analogs, HORIZON, M, TAU, THEILER
from surrogate_data_test import aaft_surrogate

ASSETS = [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20)]
N_SURROGATES = 200


def knn_atypicality(pc, j, k, theiler):
    hist = pc[:j]
    if len(hist) <= theiler + k:
        return None, None
    d = np.sum((hist - pc[j]) ** 2, axis=1)
    order = np.argsort(d)
    causal_order = [int(c) for c in order if j - int(c) > theiler]
    if len(causal_order) < max(k, M + 2):
        return None, None
    atyp = float(np.sqrt(d[causal_order[k - 1]]))
    return atyp, causal_order


def accuracy_gap_for_series(prices, log_ret, min_history=250, k_for_atyp=10):
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    origins = list(range(min_history, n_pts - HORIZON - 1, HORIZON))

    rows = []
    for j in origins:
        atyp, causal_order = knn_atypicality(pc, j, k_for_atyp, THEILER)
        if atyp is None:
            continue
        idx = causal_order[:K_ANALOGS] if len(causal_order) >= K_ANALOGS else causal_order
        p = j + (M - 1) * TAU
        if p + HORIZON >= len(prices):
            continue
        rets = [np.log(prices[ii + (M-1)*TAU + HORIZON] / prices[ii + (M-1)*TAU])
                for ii in idx if ii + (M-1)*TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred_up = np.mean(rets) > 0
        actual_up = prices[p + HORIZON] > prices[p]
        rows.append({"atyp": atyp, "correct": int(pred_up == actual_up)})

    df = pd.DataFrame(rows)
    if len(df) < 60:
        return None
    thresh = df["atyp"].median()
    hi = df.loc[df["atyp"] >= thresh, "correct"]
    lo = df.loc[df["atyp"] < thresh, "correct"]
    if len(hi) < 15 or len(lo) < 15:
        return None
    return float(hi.mean() - lo.mean())


K_ANALOGS = 7


def main():
    print(f"=== Surrogate-data test of the CORE CLAIM: atypicality-conditioned accuracy gap "
          f"(AAFT, n={N_SURROGATES}) ===\n")
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        price_df = download_max(ticker, max_years)
        prices = price_df["Close"].to_numpy(dtype=np.float64)
        log_ret = np.diff(np.log(prices))

        real_gap = accuracy_gap_for_series(prices, log_ret)
        if real_gap is None:
            print(f"  {name}: skipped (too few usable folds)")
            continue

        rng = np.random.default_rng(0)
        surrogate_gaps = []
        for s in range(N_SURROGATES):
            surr_ret = aaft_surrogate(log_ret, rng)
            # reconstruct a surrogate "price" path consistent with the surrogate returns,
            # purely so select_analogs/accuracy scoring has a well-defined price series
            surr_prices = np.concatenate([[prices[0]], prices[0] * np.exp(np.cumsum(surr_ret))])
            g = accuracy_gap_for_series(surr_prices, surr_ret)
            if g is not None:
                surrogate_gaps.append(g)
        surrogate_gaps = np.array(surrogate_gaps)

        rank_p = float(((surrogate_gaps >= real_gap).sum() + 1) / (len(surrogate_gaps) + 1))
        print(f"{name} ({time.time()-t0:.0f}s, {len(surrogate_gaps)} usable surrogates):")
        print(f"  real accuracy gap = {real_gap*100:+.2f}pp")
        print(f"  surrogate null: mean={surrogate_gaps.mean()*100:+.2f}pp  std={surrogate_gaps.std()*100:.2f}pp  "
              f"min={surrogate_gaps.min()*100:+.2f}pp  max={surrogate_gaps.max()*100:+.2f}pp")
        print(f"  rank-based one-sided p-value (real gap vs. surrogate null) = {rank_p:.4f}\n")


if __name__ == "__main__":
    main()
