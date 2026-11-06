"""
Controlled Monte Carlo sweep over pipeline hyperparameters (embedding dimension M, delay tau,
forecast horizon H, number of leaked steps) to check whether the leak-amplification artifact is
a general property of the indexing-bug class, or an artifact specific to the paper's own
hyperparameter choices (M=3, tau=1, H=3, leak=1 step).

Scope note: classical detectors only (Isolation Forest, LOF, Mahalanobis-PCA), NOT the diffusion
model, to keep per-config cost low enough to cover a real grid (diffusion retraining per
origin-fold at every one of ~100+ grid points x repeats would be computationally infeasible in
this setting). Data: synthetic Gaussian AR(1) returns (phi=0.15), matched to Brent's
unconditional volatility, 5 independent seeds per cell. This is a reduced, coarse grid, not an
exhaustive one; it is meant to establish whether the qualitative pattern (gap collapses to ~0 as
leak_steps -> 0, gap grows with |leak_steps|) holds across the grid, not to estimate precise
effect sizes per configuration.
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

N = 1800
SIGMA = 0.012
N_SEEDS = 5
MIN_HISTORY = 200


def gen_gaussian_ar1(n, sigma, seed, phi=0.15):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), size=n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return r


def prices_from_returns(log_ret, p0=100.0):
    return p0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))


def embed_param(log_ret, M, tau):
    n = len(log_ret)
    n_pts = n - (M - 1) * tau
    if n_pts < 10:
        return None
    return np.column_stack([log_ret[(M - 1 - i) * tau: (M - 1 - i) * tau + n_pts] for i in range(M)])


def run_config(M, tau, H, leak_steps, log_ret, prices, retrain_every=25):
    pc = embed_param(log_ret, M, tau)
    if pc is None:
        return None
    n_pts = len(pc)
    latest = n_pts - H - 1
    if latest <= MIN_HISTORY + 20:
        return None
    origins = list(range(MIN_HISTORY, latest, max(H, 1)))

    rows = []
    iso = lof = pca = cov = None
    for k, j in enumerate(origins):
        hist = pc[:j]
        if len(hist) < 30:
            continue
        if k % retrain_every == 0:
            iso = IsolationForest(n_estimators=60, random_state=0).fit(hist)
            lof = LocalOutlierFactor(n_neighbors=min(20, len(hist) - 1), novelty=True).fit(hist)
            n_comp = min(M, 3)
            pca = PCA(n_components=n_comp).fit(hist)
            cov = LedoitWolf().fit(pca.transform(hist))

        iso_score = -float(iso.score_samples(pc[j:j + 1])[0])
        lof_score = -float(lof.score_samples(pc[j:j + 1])[0])
        maha_score = float(cov.mahalanobis(pca.transform(pc[j:j + 1]))[0])

        # p_correct = j + M*tau (no leak); leak_steps shifts p earlier by that many steps,
        # so leak_steps=0 reproduces the fixed convention, leak_steps>=1 reproduces increasingly
        # severe versions of the original bug (the paper's own bug is leak_steps=1 under tau=1).
        # Applied uniformly to origin AND every analog's own target (see null_dgp_suite.py's
        # comment: this is what makes the leak propagate through the K-NN average at all).
        p_off = M * tau - leak_steps * tau
        p = j + p_off
        if p + H >= len(prices) or p - tau < 0 or p_off < 0:
            continue
        current_price, actual_future_price = prices[p], prices[p + H]
        nn_order = np.argsort(np.sum((hist - pc[j]) ** 2, axis=1))
        chosen = []
        for idx in nn_order:
            if idx >= j - 2:
                continue
            chosen.append(int(idx))
            if len(chosen) == 7:
                break
        if len(chosen) < 3:
            continue
        rets = [np.log(prices[idx + p_off + H] / prices[idx + p_off])
                for idx in chosen if idx + p_off + H < len(prices) and idx + p_off >= 0]
        if len(rets) < 3:
            continue
        pred_up = np.mean(rets) > 0
        actual_up = actual_future_price > current_price
        rows.append({"iso": iso_score, "lof": lof_score, "maha": maha_score,
                      "correct": int(pred_up == actual_up)})
    return pd.DataFrame(rows) if rows else None


def gap_and_p(df, col, n_perm=500, seed=0):
    vals = df[col].to_numpy()
    correct = df["correct"].to_numpy()
    med = np.median(vals)
    hi, lo = correct[vals >= med], correct[vals < med]
    if len(hi) < 5 or len(lo) < 5:
        return np.nan, np.nan
    obs = hi.mean() - lo.mean()
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        shuf = rng.permutation(vals)
        h2, l2 = correct[shuf >= med], correct[shuf < med]
        if len(h2) >= 5 and len(l2) >= 5:
            null.append(h2.mean() - l2.mean())
    null = np.array(null)
    p = float((null >= obs).mean()) if len(null) else np.nan
    return obs * 100, p


GRID = {
    "M": [2, 3, 5],
    "tau": [1, 2],
    "H": [1, 3, 5],
    "leak_steps": [0, 1, 2],
}


def main():
    results = []
    combos = [(M, tau, H, leak) for M in GRID["M"] for tau in GRID["tau"]
              for H in GRID["H"] for leak in GRID["leak_steps"]]
    print(f"Total grid cells: {len(combos)}, x {N_SEEDS} seeds = {len(combos)*N_SEEDS} runs")
    t_start = time.time()
    for ci, (M, tau, H, leak) in enumerate(combos):
        cell_gaps = {"iso": [], "lof": [], "maha": []}
        for seed in range(N_SEEDS):
            log_ret = gen_gaussian_ar1(N, SIGMA, seed=seed * 97 + ci)
            prices = prices_from_returns(log_ret)
            df = run_config(M, tau, H, leak, log_ret, prices)
            if df is None or len(df) < 30:
                continue
            for col in ("iso", "lof", "maha"):
                gap, p = gap_and_p(df, col, seed=seed)
                if not np.isnan(gap):
                    cell_gaps[col].append(gap)
        row = {"M": M, "tau": tau, "H": H, "leak_steps": leak}
        for col in ("iso", "lof", "maha"):
            row[f"{col}_mean_gap_pp"] = float(np.mean(cell_gaps[col])) if cell_gaps[col] else None
            row[f"{col}_n"] = len(cell_gaps[col])
        results.append(row)
        if ci % 10 == 0 or ci == len(combos) - 1:
            elapsed = time.time() - t_start
            print(f"[{ci+1}/{len(combos)}] M={M} tau={tau} H={H} leak={leak}  "
                  f"iso={row['iso_mean_gap_pp']}  lof={row['lof_mean_gap_pp']}  "
                  f"maha={row['maha_mean_gap_pp']}  elapsed={elapsed:.0f}s")
            sys.stdout.flush()
            with open("monte_carlo_grid_results.json", "w") as f:
                json.dump(results, f, indent=2)

    with open("monte_carlo_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Saved to monte_carlo_grid_results.json")

    df_res = pd.DataFrame(results)
    print("\n=== Mean gap by leak_steps (averaged over all M, tau, H, detectors) ===")
    for leak in GRID["leak_steps"]:
        sub = df_res[df_res["leak_steps"] == leak]
        vals = pd.concat([sub["iso_mean_gap_pp"], sub["lof_mean_gap_pp"], sub["maha_mean_gap_pp"]]).dropna()
        print(f"leak_steps={leak}: mean gap = {vals.mean():+.2f}pp (n_cells*detectors={len(vals)})")


if __name__ == "__main__":
    main()
