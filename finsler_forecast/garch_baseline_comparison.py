"""
Two things a forecasting-journal reviewer will ask after seeing finding 43 (AR(1) baseline):
A) How does the analog model compare to GARCH(1,1), the standard volatility-clustering model,
   not just a mean-return model like AR(1)?
B) Is the "atypicality" detector (finding 31/33/41) actually just re-discovering GARCH
   conditional volatility (i.e. is "atypical" a fancy synonym for "high-volatility day")? If so,
   the dynamical-systems story (finding 41) would be a less novel restatement of volatility
   clustering, and that needs to be reported honestly either way.

Method:
  A) Causal GARCH(1,1) (constant-mean + GARCH(1,1) variance, `arch` package), refit
     periodically on the same causal history as the AR(1)/analog folds, forecasting the
     cumulative H-step mean return. Same MASE-vs-random-walk evaluation as finding 42/43.
  B) For every fold, also record the GARCH model's one-step-ahead conditional volatility
     forecast, and correlate it with the diffusion atypicality score (finding 31) across all
     folds, per asset. A high correlation would mean the detector mostly tracks volatility
     clustering; a low/moderate correlation means it is capturing something distinct from GARCH
     volatility.

Result (see README Finding 44): printed output / README has the actual numbers.
"""
import os
import sys
import time
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from arch import arch_model
from economic_backtest import (HORIZON, embed, select_analogs, train_scorenet,
                                full_elbo_score, download_max, M, TAU, N_TIMESTEPS, device)
from robustness_journal_grade import ASSETS
from continuous_error_metrics import mase, rmse

warnings.filterwarnings("ignore")


def collect_with_garch(ticker, name, max_years, retrain_every=20, epochs=50, garch_refit_every=10):
    price_df = download_max(ticker, max_years)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    min_history, latest = 250, n_pts - HORIZON - 1
    origins = list(range(min_history, latest, HORIZON))

    net, schedule, garch_res, rows = None, None, None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
        if net is None or k % retrain_every == 0:
            net, schedule = train_scorenet(pc[:j+1], epochs=epochs)
        feats = net.forward(torch.tensor(pc[:j+1], dtype=torch.float32, device=device),
                             torch.full((j+1, 1), 3 / N_TIMESTEPS, device=device),
                             return_feature=True).detach().cpu().numpy()
        d_diff = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_diff = select_analogs(d_diff, j, last_ok)
        if len(idx_diff) < 3:
            continue
        diff_atyp = full_elbo_score(net, schedule, pc[j], seed=k)
        p = j + (M - 1) * TAU
        current_price, future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                for idx in idx_diff if idx + (M-1)*TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred_ret = float(np.mean(rets))
        actual_ret = float(np.log(future_price / current_price))

        # causal GARCH(1,1), percentage returns (numerical stability), refit periodically
        if garch_res is None or k % garch_refit_every == 0:
            hist_pct = log_ret[:p+1] * 100.0
            try:
                garch_res = arch_model(hist_pct, mean="Constant", vol="GARCH", p=1, q=1,
                                        rescale=False).fit(disp="off")
            except Exception:
                garch_res = None
        if garch_res is not None:
            try:
                fc = garch_res.forecast(horizon=HORIZON, reindex=False)
                garch_pred_ret = float(fc.mean.to_numpy()[-1].sum()) / 100.0
                garch_cond_vol = float(np.sqrt(fc.variance.to_numpy()[-1][0])) / 100.0
            except Exception:
                garch_pred_ret, garch_cond_vol = 0.0, np.nan
        else:
            garch_pred_ret, garch_cond_vol = 0.0, np.nan

        rows.append({"origin_idx": j, "diff_atyp": diff_atyp, "pred_ret": pred_ret,
                      "garch_pred_ret": garch_pred_ret, "garch_cond_vol": garch_cond_vol,
                      "actual_ret": actual_ret})
    return pd.DataFrame(rows)


def main():
    print("=== A) Analog model vs. GARCH(1,1) baseline | B) atypicality vs. GARCH conditional vol ===\n")
    results = []
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        df = collect_with_garch(ticker, name, max_years)
        df = df.dropna(subset=["garch_cond_vol"])
        if len(df) < 40:
            print(f"  {name}: skipped (too few folds)")
            continue

        actual = df["actual_ret"].to_numpy()
        pred_model = df["pred_ret"].to_numpy()
        pred_garch = df["garch_pred_ret"].to_numpy()

        mase_model = mase(actual, pred_model)
        mase_garch = mase(actual, pred_garch)
        corr_atyp_vol = df[["diff_atyp", "garch_cond_vol"]].corr().iloc[0, 1]

        print(f"{name:10s} (n={len(df)}, {time.time()-t0:.0f}s): "
              f"MASE analog={mase_model:.3f}  GARCH(1,1)={mase_garch:.3f}  "
              f"corr(atypicality, GARCH cond.vol)={corr_atyp_vol:+.3f}")
        results.append({"asset": name, "n": len(df), "mase_analog": mase_model,
                         "mase_garch": mase_garch, "corr_atyp_garchvol": corr_atyp_vol})

    print("\n=== Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    n_beats_garch = (res_df["mase_analog"] < res_df["mase_garch"]).sum()
    n_garch_beats_naive = (res_df["mase_garch"] < 1.0).sum()
    mean_corr = res_df["corr_atyp_garchvol"].mean()
    print(f"\n{n_beats_garch}/{len(res_df)} assets: analog model beats GARCH(1,1) (lower MASE).")
    print(f"{n_garch_beats_naive}/{len(res_df)} assets: GARCH(1,1) mean forecast beats random-walk naive.")
    print(f"Mean correlation(atypicality score, GARCH conditional volatility) across assets: {mean_corr:+.3f}")


if __name__ == "__main__":
    main()
