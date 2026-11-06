"""
A true, single-shot held-out confirmation of finding 41's dynamical mechanism (local
Lyapunov exponents), using the exact same last-20%-of-history holdout convention as finding
33-B. Finding 41 was originally tested on the full/exploratory sample, like most findings in
this investigation except the core atypicality-predictability claim (finding 33-B). Given how
many detector/architecture variants were tried over the course of this investigation, any
single additional finding built on the full sample carries real "garden of forking paths"
exposure (Gelman & Loken 2013; Coqueret 2024's "Forking paths in financial economics" makes
the same point specifically for financial anomaly research) -- so the mechanism explanation
deserves its own out-of-sample check before being presented as confirmed rather than
exploratory.

Method, deliberately mirroring finding 33-B exactly:
  1. Same 80/20 cutoff per asset, same `collect_full` fold set.
  2. The "atypical vs. typical" median threshold is learned ONLY from the pre-cutoff (training)
     period, then applied, frozen, to the held-out folds -- so no information from the holdout
     period leaks into the split criterion itself (a step finding 33-B's simpler binary-accuracy
     test didn't need, since it used the full-sample fold set's own median, but which matters
     here given how directly we are testing whether finding 41's regime-split explanation
     generalizes).
  3. Local Lyapunov exponents computed exactly as in local_lyapunov_regime.py, restricted to the
     held-out folds only.
  4. Permutation test for the mean-LLE difference (atypical vs. typical) WITHIN the held-out
     folds only.

Result (see README Finding 46): printed output / README has the actual numbers.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from economic_backtest import HORIZON, embed, download_max, M, TAU, THEILER
from robustness_journal_grade import collect_full, ASSETS
from local_lyapunov_regime import local_lyapunov_series, permutation_test_diff


def main():
    print("=== True held-out confirmation of the local-Lyapunov mechanism (finding 41) ===\n")
    results = []
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        df = collect_full(ticker, name, max_years)
        if len(df) < 60:
            print(f"  {name}: skipped (too few folds)")
            continue

        price_df_full = download_max(ticker, max_years)
        n_total = len(price_df_full)
        cutoff = int(n_total * 0.8)
        holdout_mask = df["origin_idx"] >= cutoff - (M - 1) * TAU
        train_df, holdout_df = df[~holdout_mask], df[holdout_mask].copy()
        if len(holdout_df) < 30 or len(train_df) < 30:
            print(f"  {name}: skipped (too few held-out/training folds)")
            continue

        # threshold learned ONLY from the training period, frozen, applied to holdout
        thresh = train_df["diff_atyp"].median()

        price_df = download_max(ticker, max_years)
        prices = price_df["Close"].to_numpy(dtype=np.float64)
        log_ret = np.diff(np.log(prices))
        scale = max(float(np.std(log_ret)), 1e-8)
        pc = embed(log_ret) / scale
        n_pts = len(pc)

        lle = local_lyapunov_series(pc, n_pts, holdout_df["origin_idx"].tolist())
        holdout_df["lle"] = holdout_df["origin_idx"].map(lle)
        holdout_df = holdout_df.dropna(subset=["lle"])
        if len(holdout_df) < 20:
            print(f"  {name}: skipped (too few LLE-computable held-out folds)")
            continue

        atyp_lle = holdout_df.loc[holdout_df["diff_atyp"] >= thresh, "lle"].to_numpy()
        typ_lle = holdout_df.loc[holdout_df["diff_atyp"] < thresh, "lle"].to_numpy()
        if len(atyp_lle) < 8 or len(typ_lle) < 8:
            print(f"  {name}: skipped (too few held-out folds per bucket)")
            continue
        obs, p = permutation_test_diff(atyp_lle, typ_lle)

        print(f"{name:10s} (held-out n={len(holdout_df)}, {time.time()-t0:.0f}s): "
              f"mean LLE atypical={atyp_lle.mean():+.4f}  typical={typ_lle.mean():+.4f}  "
              f"diff={obs:+.4f}  p={p:.4f}")
        results.append({"asset": name, "n_holdout": len(holdout_df), "lle_atypical": atyp_lle.mean(),
                         "lle_typical": typ_lle.mean(), "diff": obs, "p": p})

    print("\n=== Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    n_contracting_more = (res_df["diff"] < 0).sum()
    n_sig = (res_df["p"] < 0.05).sum()
    print(f"\n{n_contracting_more}/{len(res_df)} assets: held-out atypical days have LOWER "
          f"(more contracting) local Lyapunov exponent than held-out typical days.")
    print(f"{n_sig}/{len(res_df)} assets individually significant (p<0.05) on held-out data alone.")
    ps = [max(p, 1.0 / 2001) for p in res_df["p"]]
    from scipy import stats
    chi2_stat = -2 * sum(np.log(p) for p in ps)
    combined_p = 1 - stats.chi2.cdf(chi2_stat, 2 * len(ps))
    print(f"Fisher combined p-value on held-out data only: {combined_p:.6f}  ({len(ps)} assets)")


if __name__ == "__main__":
    main()
