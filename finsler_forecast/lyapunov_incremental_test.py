"""
THE decisive test, prompted directly by the surrogate-data result (findings 49): does the
local Lyapunov exponent (LLE) have any INCREMENTAL explanatory power for forecast accuracy
once the diffusion atypicality score is already controlled for -- or is the LLE-accuracy
relationship entirely mediated by (i.e., redundant with) the atypicality score itself, which
would mean the "Lyapunov mechanism" framing adds nothing beyond what finding 33 already
established?

Pre-specified decision rule (stated before running, per standard practice): fit
  correct ~ atypicality_z + LLE_z
(logistic regression, both predictors standardized) on each asset's fold data. If LLE's
coefficient is NOT significant (p > 0.05) after controlling for atypicality, the Lyapunov
finding has no incremental value and should be reported as explained by / redundant with the
atypicality score, not as an independent mechanism. If LLE remains significant, it carries
information beyond the atypicality score alone.

Result (see README Finding 50): printed output / README has the actual numbers.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from economic_backtest import embed, download_max, M, TAU
from robustness_journal_grade import collect_full, ASSETS
from local_lyapunov_regime import local_lyapunov_series


def zscore(x):
    return (x - x.mean()) / x.std()


def main():
    print("=== Decisive test: does LLE add incremental explanatory power beyond atypicality? ===\n")
    print("Pre-specified rule: LLE kept as a mechanism claim only if its logistic-regression\n"
          "coefficient remains significant (p<0.05) after controlling for the atypicality score.\n")
    results = []
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        df = collect_full(ticker, name, max_years)
        if len(df) < 60:
            print(f"  {name}: skipped (too few folds)")
            continue

        price_df = download_max(ticker, max_years)
        prices = price_df["Close"].to_numpy(dtype=np.float64)
        log_ret = np.diff(np.log(prices))
        scale = max(float(np.std(log_ret)), 1e-8)
        pc = embed(log_ret) / scale
        n_pts = len(pc)

        lle = local_lyapunov_series(pc, n_pts, df["origin_idx"].tolist())
        df["lle"] = df["origin_idx"].map(lle)
        df = df.dropna(subset=["lle"]).reset_index(drop=True)
        if len(df) < 60:
            print(f"  {name}: skipped (too few LLE-computable folds)")
            continue

        X = pd.DataFrame({"atyp_z": zscore(df["diff_atyp"]), "lle_z": zscore(df["lle"])})
        X = sm.add_constant(X)
        y = df["correct"].astype(float)
        model = sm.Logit(y, X).fit(disp=0)

        coef_atyp, p_atyp = model.params["atyp_z"], model.pvalues["atyp_z"]
        coef_lle, p_lle = model.params["lle_z"], model.pvalues["lle_z"]

        # univariate LLE-only model, for comparison (does LLE alone predict accuracy at all?)
        X_uni = sm.add_constant(pd.DataFrame({"lle_z": zscore(df["lle"])}))
        model_uni = sm.Logit(y, X_uni).fit(disp=0)
        coef_lle_uni, p_lle_uni = model_uni.params["lle_z"], model_uni.pvalues["lle_z"]

        verdict = "LLE RETAINS incremental power" if p_lle < 0.05 else "LLE REDUNDANT with atypicality"
        print(f"{name:10s} (n={len(df)}, {time.time()-t0:.0f}s):")
        print(f"  univariate:  LLE coef={coef_lle_uni:+.4f}  p={p_lle_uni:.4f}")
        print(f"  controlled:  atyp coef={coef_atyp:+.4f} p={p_atyp:.4f}   "
              f"LLE coef={coef_lle:+.4f} p={p_lle:.4f}   -> {verdict}")
        results.append({"asset": name, "n": len(df), "lle_uni_coef": coef_lle_uni,
                         "lle_uni_p": p_lle_uni, "atyp_coef": coef_atyp, "atyp_p": p_atyp,
                         "lle_coef": coef_lle, "lle_p": p_lle, "verdict": verdict})

    print("\n=== Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    n_retains = (res_df["lle_p"] < 0.05).sum()
    print(f"\n{n_retains}/{len(res_df)} assets: LLE retains significant incremental power "
          f"after controlling for the atypicality score.")


if __name__ == "__main__":
    main()
