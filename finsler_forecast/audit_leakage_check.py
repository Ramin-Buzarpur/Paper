"""
Integrity audit, prompted by an explicit request to re-check everything before targeting a
selective journal. Finding: `robustness_journal_grade.collect_full` (and the identical pattern
in `cross_asset_contagion.collect_with_dates`) fits IsolationForest/LOF/Mahalanobis-PCA on
`hist = pc[:j+1]` -- which INCLUDES today's own point -- then scores that same point. This is
in-sample ("transductive") anomaly scoring: today's point helped shape the very model used to
judge how atypical it is, every single fold, for all three classical detectors. This is subtler
than a forecast-leakage bug (it does not leak the future return used as the forecast target --
that selection is still strictly causal via `last_ok = j-1` and the Theiler window), but it is
NOT the strictly causal "fit on the past, score the present" procedure the write-up has
described. It needs to be checked, not asserted away.

This script re-runs collect_full's classical-detector scoring TWICE on the same folds for one
asset -- once as originally coded (hist = pc[:j+1], self-inclusive) and once corrected
(hist = pc[:j], strictly excluding today) -- and compares the resulting accuracy gap and
permutation p-value, to see whether the self-inclusion bias was doing any real work.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from economic_backtest import (HORIZON, embed, select_analogs, train_scorenet,
                                full_elbo_score, download_max, M, TAU, N_TIMESTEPS, device)
from statistical_rigor import permutation_test


def collect_compare(ticker, name, max_years, retrain_every=20, epochs=50):
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
        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                for idx in idx_diff if idx + (M-1)*TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred_up = np.mean(rets) > 0
        actual_up = actual_future_price > current_price
        correct = int(pred_up == actual_up)

        if j < 30:  # need a few points before "today" to fit the strictly-causal version
            continue

        # (A) ORIGINAL: self-inclusive fit (today is part of the training set)
        hist_incl = pc[:j+1]
        iso_incl = -float(IsolationForest(n_estimators=100, random_state=0).fit(hist_incl)
                           .score_samples(pc[j:j+1])[0])
        lof_incl = -float(LocalOutlierFactor(n_neighbors=30, novelty=True).fit(hist_incl)
                          .score_samples(pc[j:j+1])[0])

        # (B) CORRECTED: strictly causal fit (today excluded from training)
        hist_excl = pc[:j]
        iso_excl = -float(IsolationForest(n_estimators=100, random_state=0).fit(hist_excl)
                           .score_samples(pc[j:j+1])[0])
        lof_excl = -float(LocalOutlierFactor(n_neighbors=30, novelty=True).fit(hist_excl)
                          .score_samples(pc[j:j+1])[0])

        rows.append({"origin_idx": j, "iso_incl": iso_incl, "lof_incl": lof_incl,
                     "iso_excl": iso_excl, "lof_excl": lof_excl, "correct": correct})
    return pd.DataFrame(rows)


def report(df, col, label):
    obs, p, _ = permutation_test(df, col)
    print(f"  {label:28s}: gap={obs*100:+.1f}pp  p={p:.4f}")


def main():
    for ticker, name, max_years in [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20)]:
        t0 = time.time()
        df = collect_compare(ticker, name, max_years)
        print(f"\n=== {name} (n={len(df)}, {time.time()-t0:.0f}s) ===")
        corr_iso = df[["iso_incl", "iso_excl"]].corr().iloc[0, 1]
        corr_lof = df[["lof_incl", "lof_excl"]].corr().iloc[0, 1]
        print(f"  correlation(self-inclusive score, strictly-causal score): "
              f"IsoForest={corr_iso:.3f}  LOF={corr_lof:.3f}")
        report(df.rename(columns={"iso_incl": "atypicality"}), "atypicality", "IsoForest (self-inclusive, as coded)")
        report(df.rename(columns={"iso_excl": "atypicality"}), "atypicality", "IsoForest (strictly causal, corrected)")
        report(df.rename(columns={"lof_incl": "atypicality"}), "atypicality", "LOF (self-inclusive, as coded)")
        report(df.rename(columns={"lof_excl": "atypicality"}), "atypicality", "LOF (strictly causal, corrected)")


if __name__ == "__main__":
    main()
