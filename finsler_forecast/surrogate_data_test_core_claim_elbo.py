"""
The decisive version of surrogate_data_test_core_claim.py: repeats that test using the ACTUAL
production pipeline (diffusion-ELBO atypicality + diffusion-feature-distance analog selection,
exactly as in robustness_journal_grade.collect_full) instead of the cheap k-NN proxy, and the
exact same permutation_test convention used to report finding 33's real numbers -- to settle
whether the preliminary result in surrogate_data_test_core_claim.py (the real accuracy gap
looking weaker than the AAFT-surrogate null) survives with the real detector, or was an
artifact of that script's simplified standalone reimplementation.

Cost control: a reduced surrogate count (20) and a less frequent diffusion-net retrain cadence
(every 150 folds instead of every 20, fewer epochs) keep the full-pipeline cost per surrogate
tractable, as in surrogate_data_test_elbo.py.

Result (see README Finding 51, updated): printed output / README has the actual numbers.
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
from statistical_rigor import permutation_test
from surrogate_data_test import aaft_surrogate

ASSETS = [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20)]
N_SURROGATES = 20
RETRAIN_EVERY = 150
EPOCHS = 30


def collect_real_pipeline(prices, log_ret, min_history=250, retrain_every=RETRAIN_EVERY, epochs=EPOCHS):
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    latest = n_pts - HORIZON - 1
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
        p = j + M * TAU  # FIX (finding 51/52): was (M-1)*TAU, which leaked the future into pc[j]
        if p + HORIZON >= len(prices):
            continue
        current_price, actual_future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + M*TAU + HORIZON] / prices[idx + M*TAU])
                for idx in idx_diff if idx + M*TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred_up = np.mean(rets) > 0
        actual_up = actual_future_price > current_price
        rows.append({"atypicality": diff_atyp, "correct": int(pred_up == actual_up)})
    return pd.DataFrame(rows)


def main():
    print(f"=== DECISIVE real-pipeline surrogate test of the core claim (AAFT, n={N_SURROGATES}) ===\n")
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        price_df = download_max(ticker, max_years)
        prices = price_df["Close"].to_numpy(dtype=np.float64)
        log_ret = np.diff(np.log(prices))

        real_df = collect_real_pipeline(prices, log_ret)
        if len(real_df) < 60:
            print(f"  {name}: skipped (too few usable folds)")
            continue
        real_gap, real_p, _ = permutation_test(real_df, "atypicality")
        print(f"{name}: real gap={real_gap*100:+.2f}pp  (own permutation p={real_p:.4f})  ({time.time()-t0:.0f}s)")

        rng = np.random.default_rng(0)
        surrogate_gaps = []
        for s in range(N_SURROGATES):
            t1 = time.time()
            surr_ret = aaft_surrogate(log_ret, rng)
            surr_prices = np.concatenate([[prices[0]], prices[0] * np.exp(np.cumsum(surr_ret))])
            df = collect_real_pipeline(surr_prices, surr_ret)
            if len(df) < 60:
                print(f"  surrogate {s+1}/{N_SURROGATES}: skipped (too few folds)")
                continue
            g, _, _ = permutation_test(df, "atypicality")
            surrogate_gaps.append(g)
            print(f"  surrogate {s+1}/{N_SURROGATES}: gap={g*100:+.2f}pp  ({time.time()-t1:.0f}s)")
        surrogate_gaps = np.array(surrogate_gaps)

        rank_p = float(((surrogate_gaps >= real_gap).sum() + 1) / (len(surrogate_gaps) + 1))
        print(f"\n{name} surrogate null (n={len(surrogate_gaps)}): mean={surrogate_gaps.mean()*100:+.2f}pp "
              f"std={surrogate_gaps.std()*100:.2f}pp  min={surrogate_gaps.min()*100:+.2f}pp  "
              f"max={surrogate_gaps.max()*100:+.2f}pp")
        print(f"rank-based one-sided p-value (real gap vs. surrogate null) = {rank_p:.4f}\n")


if __name__ == "__main__":
    main()
