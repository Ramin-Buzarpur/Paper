"""
Every finding so far reports BINARY directional accuracy (predicted up/down vs. actual
up/down). Standard forecasting practice calls for continuous forecast-error metrics
(MASE, RMSE) benchmarked against a naive forecast -- directional accuracy alone invites the
objection "a coin flip has 50% accuracy too; how much of the RETURN did you actually get
right, and against what baseline?"

This script re-runs the same causal, diffusion-metric analog-selection forecast as
robustness_journal_grade.collect_full, but keeps the CONTINUOUS predicted and actual H-day log
returns (not just their signs), and evaluates:
  - RMSE of the model's forecast vs. the actual return.
  - MASE (Mean Absolute Scaled Error, Hyndman & Koehler 2006 -- the standard forecasting-
    journal metric): MAE(model) / MAE(naive), where the naive benchmark is the canonical
    "no-change" / random-walk forecast (predicted return = 0), appropriate for log returns.
    MASE < 1 means the model beats the random-walk benchmark; MASE > 1 means it's worse.
  - The same comparison split by the diffusion atypicality gate (median split, same convention
    as every other finding here): does the continuous forecast error -- not just the binary
    direction -- also improve on flagged "atypical" days?

Result (see README Finding 42): printed output / README has the actual numbers.
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
from robustness_journal_grade import ASSETS


def collect_continuous(ticker, name, max_years, retrain_every=20, epochs=50):
    price_df = download_max(ticker, max_years)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    min_history, latest = 250, n_pts - HORIZON - 1
    origins = list(range(min_history, latest, HORIZON))

    net, schedule, rows = None, None, []
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
        rows.append({"origin_idx": j, "diff_atyp": diff_atyp, "pred_ret": pred_ret,
                     "actual_ret": actual_ret})
    return pd.DataFrame(rows)


def mase(actual, pred, naive_pred=0.0):
    mae_model = np.mean(np.abs(actual - pred))
    mae_naive = np.mean(np.abs(actual - naive_pred))
    return mae_model / max(mae_naive, 1e-12)


def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def bootstrap_compare(actual, pred_model, pred_other, n_boot=2000, seed=0):
    """Paired bootstrap over folds (resampled with replacement) comparing mean absolute error
    of `pred_model` vs. a baseline `pred_other` (random-walk naive, AR(1), or GARCH(1,1)).
    Returns (mean MAE-improvement, 95% CI lo, 95% CI hi, one-sided p-value for H1: model's MAE
    is lower). Positive improvement / CI above 0 / small p all support "model beats baseline"."""
    rng = np.random.default_rng(seed)
    n = len(actual)
    err_model = np.abs(actual - pred_model)
    err_other = np.abs(actual - pred_other)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = err_other[idx].mean() - err_model[idx].mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_one_sided = float(((diffs <= 0).sum() + 1) / (n_boot + 1))
    return float(diffs.mean()), float(ci_lo), float(ci_hi), p_one_sided


def main():
    print("=== Continuous forecast-error evaluation (RMSE, MASE vs. random-walk naive) ===\n")
    results = []
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        df = collect_continuous(ticker, name, max_years)
        if len(df) < 40:
            print(f"  {name}: skipped (too few folds)")
            continue

        actual, pred = df["actual_ret"].to_numpy(), df["pred_ret"].to_numpy()
        overall_mase = mase(actual, pred)
        overall_rmse = rmse(actual, pred)
        naive_rmse = rmse(actual, np.zeros_like(actual))

        thresh = df["diff_atyp"].median()
        hi = df["diff_atyp"] >= thresh
        lo = df["diff_atyp"] < thresh
        mase_hi = mase(df.loc[hi, "actual_ret"].to_numpy(), df.loc[hi, "pred_ret"].to_numpy())
        mase_lo = mase(df.loc[lo, "actual_ret"].to_numpy(), df.loc[lo, "pred_ret"].to_numpy())

        imp, ci_lo, ci_hi, p = bootstrap_compare(actual, pred, np.zeros_like(actual))

        print(f"{name:10s} (n={len(df)}, {time.time()-t0:.0f}s): "
              f"RMSE model={overall_rmse:.5f} vs naive={naive_rmse:.5f} | "
              f"MASE overall={overall_mase:.3f} | MASE atypical={mase_hi:.3f} typical={mase_lo:.3f} | "
              f"MAE-improvement vs naive={imp:+.5f} 95%CI=[{ci_lo:+.5f},{ci_hi:+.5f}] p={p:.4f}")
        results.append({"asset": name, "n": len(df), "rmse_model": overall_rmse,
                         "rmse_naive": naive_rmse, "mase_overall": overall_mase,
                         "mase_atypical": mase_hi, "mase_typical": mase_lo,
                         "mae_improvement": imp, "ci_lo": ci_lo, "ci_hi": ci_hi, "boot_p": p})

    print("\n=== Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    n_beats_naive = (res_df["mase_overall"] < 1.0).sum()
    n_atyp_better = (res_df["mase_atypical"] < res_df["mase_typical"]).sum()
    print(f"\n{n_beats_naive}/{len(res_df)} assets: model beats random-walk naive overall (MASE<1).")
    print(f"{n_atyp_better}/{len(res_df)} assets: continuous forecast error is lower on atypical "
          f"days than typical days (MASE_atypical < MASE_typical).")


if __name__ == "__main__":
    main()
