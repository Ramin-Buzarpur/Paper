"""
A calendar-subperiod robustness check for both headline, holdout-confirmed claims: is the
atypicality-predictability link (finding 33) and the local-Lyapunov mechanism (finding 41/46)
a stable stylized fact across different eras of market history, or mostly a product of one
particular period (e.g. the 2020-2022 COVID/rate-hike stretch)? This is the natural follow-up
question after finding 47's parameter-sensitivity check: that check varied the ESTIMATOR's own
parameters; this one varies the TIME PERIOD the data comes from.

Method: split each asset's full fold history at its calendar midpoint into an "early" half and
a "late" half (not a random split -- a genuine calendar split, so each half corresponds to a
materially different macro era for most of these assets), and re-run both the binary-accuracy
permutation test and the local-Lyapunov permutation test independently within each half.

Result (see README Finding 48): printed output / README has the actual numbers.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from economic_backtest import embed, download_max, M, TAU
from robustness_journal_grade import collect_full, ASSETS
from statistical_rigor import permutation_test
from local_lyapunov_regime import local_lyapunov_series, permutation_test_diff


def main():
    print("=== Calendar-subperiod robustness: early half vs. late half of each asset's history ===\n")
    results = []
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        df = collect_full(ticker, name, max_years)
        if len(df) < 80:
            print(f"  {name}: skipped (too few folds)")
            continue

        price_df = download_max(ticker, max_years)
        prices = price_df["Close"].to_numpy(dtype=np.float64)
        log_ret = np.diff(np.log(prices))
        scale = max(float(np.std(log_ret)), 1e-8)
        pc = embed(log_ret) / scale
        n_pts = len(pc)

        mid = df["origin_idx"].median()
        early = df[df["origin_idx"] < mid].reset_index(drop=True)
        late = df[df["origin_idx"] >= mid].reset_index(drop=True)

        start_date = price_df.index[int(df["origin_idx"].min())].date()
        mid_date = price_df.index[int(mid)].date()
        end_date = price_df.index[int(df["origin_idx"].max())].date()

        row = {"asset": name, "n_early": len(early), "n_late": len(late),
               "start": start_date, "mid": mid_date, "end": end_date}

        for label, half in [("early", early), ("late", late)]:
            if len(half) < 30:
                row[f"acc_gap_{label}"], row[f"acc_p_{label}"] = np.nan, np.nan
                row[f"lle_diff_{label}"], row[f"lle_p_{label}"] = np.nan, np.nan
                continue
            obs, p, _ = permutation_test(half, "diff_atyp")
            row[f"acc_gap_{label}"], row[f"acc_p_{label}"] = obs, p

            lle = local_lyapunov_series(pc, n_pts, half["origin_idx"].tolist())
            h = half.copy()
            h["lle"] = h["origin_idx"].map(lle)
            h = h.dropna(subset=["lle"])
            if len(h) < 20:
                row[f"lle_diff_{label}"], row[f"lle_p_{label}"] = np.nan, np.nan
                continue
            thresh = h["diff_atyp"].median()
            atyp = h.loc[h["diff_atyp"] >= thresh, "lle"].to_numpy()
            typ = h.loc[h["diff_atyp"] < thresh, "lle"].to_numpy()
            if len(atyp) < 8 or len(typ) < 8:
                row[f"lle_diff_{label}"], row[f"lle_p_{label}"] = np.nan, np.nan
                continue
            lle_obs, lle_p = permutation_test_diff(atyp, typ)
            row[f"lle_diff_{label}"], row[f"lle_p_{label}"] = lle_obs, lle_p

        print(f"{name:10s} ({start_date} .. {mid_date} .. {end_date}, {time.time()-t0:.0f}s)")
        print(f"  accuracy gap   early(n={len(early):4d})={row.get('acc_gap_early', float('nan'))*100:+5.1f}pp p={row.get('acc_p_early', float('nan')):.4f}   "
              f"late(n={len(late):4d})={row.get('acc_gap_late', float('nan'))*100:+5.1f}pp p={row.get('acc_p_late', float('nan')):.4f}")
        print(f"  LLE difference early={row.get('lle_diff_early', float('nan')):+.4f} p={row.get('lle_p_early', float('nan')):.4f}   "
              f"late={row.get('lle_diff_late', float('nan')):+.4f} p={row.get('lle_p_late', float('nan')):.4f}")
        results.append(row)

    print("\n=== Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
