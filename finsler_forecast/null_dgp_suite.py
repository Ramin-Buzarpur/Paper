"""
Extends the paper's white-noise null control (Gaussian IID) to three additional null DGPs that
real financial returns actually resemble, to check whether the leak-plus-anomaly-conditioning
artifact is specific to the Gaussian-IID case or general across null processes with fatter
tails / serial dependence / volatility clustering:

  1. Student-t IID        (heavy tails, no dependence)
  2. Gaussian AR(1)        (linear serial dependence, no heavy tails)
  3. GARCH(1,1)            (volatility clustering, conditionally heteroskedastic)

For each DGP, both the BUGGY (p = j + (M-1)*TAU) and FIXED (p = j + M*TAU) index convention are
run through the identical diffusion-ELBO pipeline, across N_SEEDS independent draws, matched in
sample length and unconditional volatility to Brent. A real accuracy-gap artifact under the bug
that disappears under the fix, on every one of these null processes, would show the artifact is
a property of the indexing bug itself, not of the Gaussian-IID assumption specifically.
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from economic_backtest import (HORIZON, embed, select_analogs, train_scorenet,
                                full_elbo_score, download_max, M, TAU, N_TIMESTEPS, device)
from statistical_rigor import permutation_test

N = 2500          # matched to a typical ~10-20yr daily series length
SIGMA = 0.012      # roughly matched to Brent's daily return vol
N_SEEDS = 5
MIN_HISTORY = 250


def gen_student_t(n, sigma, seed, df=4):
    rng = np.random.default_rng(seed)
    x = rng.standard_t(df, size=n)
    x *= sigma / x.std()
    return x


def gen_gaussian_ar1(n, sigma, seed, phi=0.15):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), size=n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return r


def gen_garch11(n, sigma, seed, omega=None, alpha=0.08, beta=0.90):
    rng = np.random.default_rng(seed)
    if omega is None:
        omega = sigma ** 2 * (1 - alpha - beta)
    r = np.zeros(n)
    h = sigma ** 2
    for t in range(n):
        h = omega + alpha * (r[t - 1] ** 2 if t > 0 else sigma ** 2) + beta * h
        r[t] = rng.normal(0, np.sqrt(h))
    return r


def prices_from_returns(log_ret, p0=100.0):
    return p0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))


def collect(prices, log_ret, buggy, retrain_every=20, epochs=30):
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    latest = n_pts - HORIZON - 1
    origins = list(range(MIN_HISTORY, latest, HORIZON))

    net, schedule, rows = None, None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
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

        # the index convention is a property of the pipeline applied uniformly to every origin,
        # not a special case for the one being forecast -- so analogs' own realized targets use
        # the identical (buggy or fixed) convention as the origin being scored, matching how the
        # real bug actually manifested (and matching the neighbor-propagation mechanism in the
        # paper's Section 4.4: the leak reaches the forecast through every analog's own target).
        p_off = (M - 1) * TAU if buggy else M * TAU
        p = j + p_off
        if p + HORIZON >= len(prices) or p < 0:
            continue
        current_price, actual_future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + p_off + HORIZON] / prices[idx + p_off])
                for idx in idx_diff if idx + p_off + HORIZON < len(prices) and idx + p_off >= 0]
        if len(rets) < 3:
            continue
        pred_up = np.mean(rets) > 0
        actual_up = actual_future_price > current_price
        rows.append({"atypicality": diff_atyp, "correct": int(pred_up == actual_up)})
    return pd.DataFrame(rows)


DGPS = {
    "student_t(df=4)": gen_student_t,
    "gaussian_ar1(phi=0.15)": gen_gaussian_ar1,
    "garch11(alpha=.08,beta=.90)": gen_garch11,
}


def main():
    results = {}
    for dgp_name, gen in DGPS.items():
        results[dgp_name] = {"buggy": [], "fixed": []}
        for seed in range(N_SEEDS):
            log_ret = gen(N, SIGMA, seed)
            prices = prices_from_returns(log_ret)
            for buggy in (True, False):
                t0 = time.time()
                df = collect(prices, log_ret, buggy=buggy)
                if len(df) < 30:
                    print(f"{dgp_name} seed={seed} buggy={buggy}: skipped n={len(df)}")
                    continue
                gap, p_val, _ = permutation_test(df, "atypicality")
                key = "buggy" if buggy else "fixed"
                results[dgp_name][key].append({"seed": seed, "gap_pp": gap * 100, "p": p_val, "n": len(df)})
                print(f"{dgp_name:30s} seed={seed} {key:6s} n={len(df):4d} "
                      f"gap={gap*100:+6.2f}pp p={p_val:.4f} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()
        with open("null_dgp_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Summary (mean gap across seeds) ===")
    for dgp_name in DGPS:
        for key in ("buggy", "fixed"):
            vals = [r["gap_pp"] for r in results[dgp_name][key]]
            if vals:
                print(f"{dgp_name:30s} {key:6s} mean_gap={np.mean(vals):+6.2f}pp "
                      f"(n_seeds={len(vals)}, sd={np.std(vals):.2f})")


if __name__ == "__main__":
    main()
