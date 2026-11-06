"""
A parameter-sensitivity check for the local-Lyapunov mechanism (findings 41/46) -- the single
most important remaining robustness question a careful reviewer would ask about the paper's
headline mechanism: is the "atypical days are locally more contracting" result an artifact of
the specific Theiler-exclusion window and divergence horizon chosen, or does it hold under
reasonable alternative choices?

Method: for Brent (the strongest asset) and S&P 500 (the persistently weakest asset -- the
hardest case to pass), recompute the local Lyapunov exponent under 3x3 = 9 combinations of:
  - Theiler window: 5, 10 (the value used everywhere else in this investigation), 20
  - divergence horizon: 3 (= HORIZON, used everywhere else), 6, 10 steps
...reusing the SAME diffusion-atypicality fold data (collect_full) computed once per asset, so
only the Lyapunov-estimation step itself varies across the 9x2 = 18 conditions.

Result (see README Finding 47): printed output / README has the actual numbers.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from economic_backtest import embed, download_max, M, TAU
from robustness_journal_grade import collect_full
from local_lyapunov_regime import permutation_test_diff

ASSETS = [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20)]
THEILER_VALUES = [5, 10, 20]
HORIZON_VALUES = [3, 6, 10]


def nearest_neighbor_causal(pc, j, theiler):
    hist = pc[:j]
    if len(hist) <= theiler + 1:
        return None
    d = np.sum((hist - pc[j]) ** 2, axis=1)
    order = np.argsort(d)
    for cand in order:
        if j - int(cand) > theiler:
            return int(cand)
    return None


def local_lyapunov_series(pc, n_pts, origin_idxs, horizon, theiler):
    out = {}
    for j in origin_idxs:
        nn = nearest_neighbor_causal(pc, j, theiler)
        if nn is None:
            continue
        j_f, nn_f = j + horizon, nn + horizon
        if j_f >= n_pts or nn_f >= n_pts:
            continue
        d_now = np.linalg.norm(pc[j] - pc[nn])
        d_future = np.linalg.norm(pc[j_f] - pc[nn_f])
        if d_now < 1e-10:
            continue
        out[j] = float(np.log(max(d_future, 1e-10) / d_now) / horizon)
    return out


def main():
    print("=== Parameter-sensitivity check for the local-Lyapunov mechanism (Theiler x horizon) ===\n")
    results = []
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        df = collect_full(ticker, name, max_years)
        print(f"{name}: {len(df)} folds collected in {time.time()-t0:.0f}s\n")

        price_df = download_max(ticker, max_years)
        prices = price_df["Close"].to_numpy(dtype=np.float64)
        log_ret = np.diff(np.log(prices))
        scale = max(float(np.std(log_ret)), 1e-8)
        pc = embed(log_ret) / scale
        n_pts = len(pc)
        origins = df["origin_idx"].tolist()

        for theiler in THEILER_VALUES:
            for horizon in HORIZON_VALUES:
                lle = local_lyapunov_series(pc, n_pts, origins, horizon, theiler)
                sub = df.copy()
                sub["lle"] = sub["origin_idx"].map(lle)
                sub = sub.dropna(subset=["lle"])
                if len(sub) < 30:
                    continue
                thresh = sub["diff_atyp"].median()
                atyp = sub.loc[sub["diff_atyp"] >= thresh, "lle"].to_numpy()
                typ = sub.loc[sub["diff_atyp"] < thresh, "lle"].to_numpy()
                obs, p = permutation_test_diff(atyp, typ, n_perm=1000)
                print(f"  {name:8s} theiler={theiler:2d} horizon={horizon:2d}: "
                      f"diff={obs:+.4f}  p={p:.4f}  (n={len(sub)})")
                results.append({"asset": name, "theiler": theiler, "horizon": horizon,
                                 "diff": obs, "p": p, "n": len(sub)})
        print()

    print("=== Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    n_consistent = (res_df["diff"] < 0).sum()
    n_sig = (res_df["p"] < 0.05).sum()
    print(f"\n{n_consistent}/{len(res_df)} parameter combinations: direction holds (atypical < typical LLE).")
    print(f"{n_sig}/{len(res_df)} parameter combinations: individually significant (p<0.05).")


if __name__ == "__main__":
    main()
