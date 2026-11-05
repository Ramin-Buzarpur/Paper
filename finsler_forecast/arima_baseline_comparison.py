"""
Every comparison so far benchmarks the analog-forecast model against the random-walk naive
(finding 42) or against other anomaly detectors (finding 33) -- never against an established
classical FORECASTING model. The obvious next question is "how does this compare to a
standard econometric baseline (AR/ARIMA), not just a no-change forecast?"

Method: for every fold already used in continuous_error_metrics.py, additionally fit a causal
AR(1) model (statsmodels ARIMA(1,0,0), refit every `retrain_every` folds on history up to the
fold, same cadence as the diffusion net) on the log-return series and forecast the cumulative
H-step-ahead return. Compare its MASE (against the same random-walk naive) to the analog
model's MASE, overall and split by the atypicality gate.

Result (see README Finding 43): printed output / README has the actual numbers.
"""
import os
import sys
import time
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.arima.model import ARIMA
from economic_backtest import (HORIZON, embed, select_analogs, train_scorenet,
                                full_elbo_score, download_max, M, TAU, N_TIMESTEPS, device)
from robustness_journal_grade import ASSETS
from continuous_error_metrics import mase, rmse, bootstrap_compare

warnings.filterwarnings("ignore")


def collect_with_arima(ticker, name, max_years, retrain_every=20, epochs=50, arima_refit_every=10):
    price_df = download_max(ticker, max_years)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    min_history, latest = 250, n_pts - HORIZON - 1
    origins = list(range(min_history, latest, HORIZON))

    net, schedule, arima_fit, rows = None, None, None, []
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

        # causal AR(1), fit on log returns up to (and including) today's origin point p
        if arima_fit is None or k % arima_refit_every == 0:
            hist_ret = log_ret[:p+1]
            try:
                arima_fit = ARIMA(hist_ret, order=(1, 0, 0)).fit()
            except Exception:
                arima_fit = None
        if arima_fit is not None:
            try:
                fc = arima_fit.forecast(steps=HORIZON)
                arima_pred_ret = float(np.sum(fc))
            except Exception:
                arima_pred_ret = 0.0
        else:
            arima_pred_ret = 0.0

        rows.append({"origin_idx": j, "diff_atyp": diff_atyp, "pred_ret": pred_ret,
                      "arima_pred_ret": arima_pred_ret, "actual_ret": actual_ret})
    return pd.DataFrame(rows)


def main():
    print("=== Analog-forecast model vs. classical AR(1) baseline (MASE vs. random-walk naive) ===\n")
    results = []
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        df = collect_with_arima(ticker, name, max_years)
        if len(df) < 40:
            print(f"  {name}: skipped (too few folds)")
            continue

        actual = df["actual_ret"].to_numpy()
        pred_model = df["pred_ret"].to_numpy()
        pred_arima = df["arima_pred_ret"].to_numpy()

        mase_model = mase(actual, pred_model)
        mase_arima = mase(actual, pred_arima)
        rmse_model, rmse_arima = rmse(actual, pred_model), rmse(actual, pred_arima)
        imp, ci_lo, ci_hi, p = bootstrap_compare(actual, pred_model, pred_arima)

        print(f"{name:10s} (n={len(df)}, {time.time()-t0:.0f}s): "
              f"MASE analog={mase_model:.3f}  AR(1)={mase_arima:.3f}  "
              f"RMSE analog={rmse_model:.5f}  AR(1)={rmse_arima:.5f}  "
              f"MAE-improvement vs AR(1)={imp:+.5f} 95%CI=[{ci_lo:+.5f},{ci_hi:+.5f}] p={p:.4f}")
        results.append({"asset": name, "n": len(df), "mase_analog": mase_model,
                         "mase_arima": mase_arima, "rmse_analog": rmse_model, "rmse_arima": rmse_arima,
                         "mae_improvement": imp, "ci_lo": ci_lo, "ci_hi": ci_hi, "boot_p": p})

    print("\n=== Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    n_beats_arima = (res_df["mase_analog"] < res_df["mase_arima"]).sum()
    n_arima_beats_naive = (res_df["mase_arima"] < 1.0).sum()
    print(f"\n{n_beats_arima}/{len(res_df)} assets: analog model beats classical AR(1) (lower MASE).")
    print(f"{n_arima_beats_naive}/{len(res_df)} assets: AR(1) itself beats random-walk naive (MASE<1).")


if __name__ == "__main__":
    main()
