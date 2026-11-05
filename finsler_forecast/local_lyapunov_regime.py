"""
A theoretical mechanism for WHY atypical market states are more predictable, grounded in
dynamical-systems/control theory rather than just an empirical correlation between detector
scores and accuracy.

Motivation (literature): local/finite-time Lyapunov exponents (LLE) generalize the classical
(global) Lyapunov exponent -- the rate at which nearby trajectories in a dynamical system's
phase space diverge -- to a state-dependent, time-varying quantity. Even in a globally chaotic
system, there exist localized "pockets" of negative (contracting) LLE where nearby
trajectories temporarily converge rather than diverge, and these pockets are, by construction,
more predictable in the short run (Kantz & Schreiber; Yao & Tong on "prediction of
predictability"; several recent applications to crash indicators and crypto chaos). This is
the same mathematical object (rate of state-space contraction/expansion) used throughout
nonlinear control theory to certify local stability -- Lyapunov's direct method.

This script tests whether that mechanism is actually present in our delay-embedded return
space: are the "atypical" days flagged by the anomaly detectors (finding 31/33) systematically
associated with locally CONTRACTING dynamics (negative local Lyapunov exponent), which would
give a first-principles dynamical explanation for the core empirical finding, rather than
leaving it as an unexplained correlation.

Method (causal where it matters, diagnostic elsewhere):
  1. Reuse the exact same causal fold structure and diffusion-atypicality scores as
     robustness_journal_grade.collect_full (no new leakage introduced into the forecast itself).
  2. For each fold's origin day j, find its nearest neighbor in the delay-embedded phase space
     (Theiler-windowed, same convention as the core forecasting method) using ONLY history up to
     j -- this nearest-neighbor pair defines two nearby trajectories, exactly as in the
     Rosenstein/Wolf method for estimating Lyapunov exponents from observed data.
  3. Track how far apart these two trajectories are HORIZON steps later and compute the local
     Lyapunov exponent estimate: LLE_j = (1/HORIZON) * ln(d_future / d_now).
     NOTE: this quantity itself requires future data (same as the standard Rosenstein estimator)
     and is used here purely as a RETROSPECTIVE, explanatory diagnostic of the market's local
     dynamics -- not as a new trading signal or forecasting input. It explains the finding; it
     does not re-derive it with leakage.
  4. Compare the mean LLE on days the diffusion/classical detectors flagged "atypical" vs.
     "typical" (median split, same convention as every other finding in this investigation),
     per asset, with a permutation test for the difference in means.

Result (see README Finding 41): see printed output / README for the numbers actually observed.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from economic_backtest import HORIZON, embed, download_max, M, TAU, THEILER
from robustness_journal_grade import collect_full, ASSETS


def nearest_neighbor_causal(pc, j, theiler=THEILER):
    """Nearest neighbor to point j using only indices < j (causal), Theiler-excluded."""
    hist = pc[:j]
    if len(hist) <= theiler + 1:
        return None
    d = np.sum((hist - pc[j]) ** 2, axis=1)
    order = np.argsort(d)
    for cand in order:
        if j - int(cand) > theiler:
            return int(cand)
    return None


def local_lyapunov_series(pc, n_pts, origin_idxs, horizon=HORIZON, theiler=THEILER):
    """LLE_j = (1/H) * ln(||pc[j+H]-pc[nn+H]|| / ||pc[j]-pc[nn]||) for each origin index."""
    out = {}
    for j in origin_idxs:
        nn = nearest_neighbor_causal(pc, j, theiler)
        if nn is None:
            continue
        j_f, nn_f = j + horizon, nn + horizon
        if j_f >= n_pts or nn_f >= n_pts:
            continue
        d_now = np.linalg.norm(pc[j] - pc[nn])
        d_future = np.linalg.norm(pc[j_f] - pc[nn_f])
        if d_now < 1e-10:
            continue
        out[j] = float(np.log(max(d_future, 1e-10) / d_now) / horizon)
    return out


def permutation_test_diff(a, b, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])
        if abs(diff) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    print("=== Local Lyapunov exponent as a dynamical-systems explanation of regime predictability ===\n")
    results = []
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        try:
            df = collect_full(ticker, name, max_years)
        except Exception as e:
            print(f"  {name}: skipped ({e})")
            continue
        if len(df) < 40:
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
        df = df.dropna(subset=["lle"])
        if len(df) < 30:
            print(f"  {name}: skipped (too few LLE-computable folds)")
            continue

        thresh = df["diff_atyp"].median()
        atyp_lle = df.loc[df["diff_atyp"] >= thresh, "lle"].to_numpy()
        typ_lle = df.loc[df["diff_atyp"] < thresh, "lle"].to_numpy()
        obs, p = permutation_test_diff(atyp_lle, typ_lle)

        corr = df[["diff_atyp", "lle"]].corr().iloc[0, 1]
        print(f"{name:10s} (n={len(df)}, {time.time()-t0:.0f}s): "
              f"mean LLE atypical={atyp_lle.mean():+.4f}  typical={typ_lle.mean():+.4f}  "
              f"diff={obs:+.4f}  p={p:.4f}  corr(atyp,LLE)={corr:+.3f}")
        results.append({"asset": name, "lle_atypical": atyp_lle.mean(), "lle_typical": typ_lle.mean(),
                         "diff": obs, "p": p, "corr": corr, "n": len(df)})

    print("\n=== Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    n_contracting_more = (res_df["diff"] < 0).sum()
    print(f"\n{n_contracting_more}/{len(res_df)} assets: atypical days have LOWER (more contracting) "
          f"local Lyapunov exponent than typical days.")


if __name__ == "__main__":
    main()
