"""
Full 28-cell extension of leakage_audit_checklist.py's check 2 (which only ran Brent/S&P500,
diffusion-ELBO only). This runs all 7 assets x all 4 detectors, comparing the full-sample
normalization actually used in the paper against a causal (per-origin, growing-window) scale,
and reports the Spearman correlation between each detector's atypicality score and the leaked
return's magnitude |log_ret[p]| as direct empirical evidence for the Section 4.4 mechanism
claim (every detector should show a positive correlation, not just those that are provably
norm-monotonic).

Output: prints a full table; also writes JSON to causal_norm_results.json so the paper-writing
step doesn't have to re-parse stdout.
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from economic_backtest import (HORIZON, embed, select_analogs, train_scorenet,
                                full_elbo_score, download_max, M, TAU, N_TIMESTEPS, device)
from statistical_rigor import permutation_test

ASSETS = [
    ("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20), ("GC=F", "Gold", 20),
    ("BTC-USD", "Bitcoin", 10), ("ETH-USD", "Ethereum", 8),
    ("SI=F", "Silver", 20), ("CL=F", "WTI Crude", 20),
]


def collect(ticker, name, max_years, causal_scale, retrain_every=20, epochs=50):
    price_df = download_max(ticker, max_years)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    global_scale = max(float(np.std(log_ret)), 1e-8)
    n_pts_full = len(log_ret) - (M - 1) * TAU
    min_history, latest = 250, n_pts_full - HORIZON - 1
    origins = list(range(min_history, latest, HORIZON))

    net, schedule, rows = None, None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
        scale = max(float(np.std(log_ret[:j + M * TAU + 1])), 1e-8) if causal_scale else global_scale
        pc = embed(log_ret) / scale

        if net is None or k % retrain_every == 0:
            net, schedule = train_scorenet(pc[:j + 1], epochs=epochs)
        feats = net.forward(torch.tensor(pc[:j + 1], dtype=torch.float32, device=device),
                             torch.full((j + 1, 1), 3 / N_TIMESTEPS, device=device),
                             return_feature=True).detach().cpu().numpy()
        d_diff = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_diff = select_analogs(d_diff, j, last_ok)
        if len(idx_diff) < 3:
            continue
        diff_atyp = full_elbo_score(net, schedule, pc[j], seed=k)

        p = j + M * TAU
        if p + HORIZON >= len(prices):
            continue
        current_price, actual_future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + M * TAU + HORIZON] / prices[idx + M * TAU])
                for idx in idx_diff if idx + M * TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred_up = np.mean(rets) > 0
        actual_up = actual_future_price > current_price

        hist = pc[:j]
        iso = IsolationForest(n_estimators=100, random_state=0).fit(hist)
        iso_score = -float(iso.score_samples(pc[j:j + 1])[0])
        lof = LocalOutlierFactor(n_neighbors=30, novelty=True).fit(hist)
        lof_score = -float(lof.score_samples(pc[j:j + 1])[0])
        pca = PCA(n_components=3).fit(hist)
        z = pca.transform(hist)
        cov = LedoitWolf().fit(z)
        maha_score = float(cov.mahalanobis(pca.transform(pc[j:j + 1]))[0])

        # embedding's own zero-lag component, log_ret[j+(M-1)*TAU] -- literally pc[j][0] before
        # scaling. This is what the BUGGY pipeline reads as log_ret[p_buggy] and feeds straight
        # into the target; it is NOT the same as log_ret[p] (the FIXED target's first step,
        # one step later), which is what an earlier version of this script used by mistake --
        # that measured ordinary volatility clustering between adjacent days, not the detector's
        # sensitivity to its own leaked input.
        leak_mag = abs(float(log_ret[j + (M - 1) * TAU]))

        rows.append({
            "origin_idx": j, "diff_atyp": diff_atyp, "iso": iso_score, "lof": lof_score,
            "maha": maha_score, "correct": int(pred_up == actual_up),
            "leak_mag": leak_mag,
        })
    return pd.DataFrame(rows)


def main():
    results = {"causal_norm": {}, "spearman": {}}
    detectors = [("diff_atyp", "Diffusion"), ("iso", "IsolationForest"),
                 ("lof", "LOF"), ("maha", "Mahalanobis-PCA")]

    for ticker, name, max_years in ASSETS:
        for causal in (False, True):
            label = "causal" if causal else "global"
            t0 = time.time()
            df = collect(ticker, name, max_years, causal_scale=causal)
            if len(df) < 30:
                print(f"{name} ({label}): skipped, n={len(df)}")
                continue
            row = {"n": len(df)}
            for col, dname in detectors:
                gap, p_val, _ = permutation_test(df, col)
                rho, rho_p = spearmanr(df[col], df["leak_mag"])
                row[dname] = {"gap_pp": gap * 100, "p": p_val, "spearman_rho": rho, "spearman_p": rho_p}
            results["causal_norm"].setdefault(name, {})[label] = row
            print(f"{name} ({label}, n={len(df)}, {time.time()-t0:.0f}s):")
            for col, dname in detectors:
                r = row[dname]
                print(f"  {dname:16s} gap={r['gap_pp']:+6.2f}pp p={r['p']:.4f}  "
                      f"Spearman(rho,|leak|)={r['spearman_rho']:+.3f} (p={r['spearman_p']:.4g})")
            sys.stdout.flush()

    with open("causal_norm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to causal_norm_results.json")


if __name__ == "__main__":
    main()
