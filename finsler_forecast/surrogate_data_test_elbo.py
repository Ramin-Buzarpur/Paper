"""
Closes the methodological gap explicitly flagged after finding 49: that surrogate-data test
used a cheap k-NN distance proxy for atypicality (for computational tractability across 200
surrogates), not the actual diffusion-ELBO score used in findings 41/46/47/48. This script
repeats the same AAFT surrogate-data test with the REAL diffusion ELBO atypicality score, on
S&P 500 only, with a reduced surrogate count (20) and a less frequent diffusion-net retrain
cadence (every 150 folds instead of every 20) to keep the full-pipeline cost per surrogate
tractable -- this is a reduced-power confirmatory check, not a replacement for finding 49's
larger-sample kNN version; its purpose is only to verify finding 49's conclusion does not
depend on which atypicality detector is used, consistent with finding 33's "detector-agnostic"
result.

Result (see README Finding 49, updated): printed output / README has the actual numbers.
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
from local_lyapunov_regime import local_lyapunov_series
from surrogate_data_test import aaft_surrogate

N_SURROGATES = 20
RETRAIN_EVERY = 150
EPOCHS = 30


def elbo_atypicality_and_lle(log_ret, min_history=250):
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    origins = list(range(min_history, n_pts - HORIZON - 1, HORIZON))

    net, schedule, atyp = None, None, {}
    for k, j in enumerate(origins):
        if net is None or k % RETRAIN_EVERY == 0:
            net, schedule = train_scorenet(pc[:j+1], epochs=EPOCHS)
        atyp[j] = full_elbo_score(net, schedule, pc[j], seed=k)

    lle = local_lyapunov_series(pc, n_pts, origins)
    common = [j for j in origins if j in atyp and j in lle]
    if len(common) < 60:
        return None
    a = np.array([atyp[j] for j in common])
    l = np.array([lle[j] for j in common])
    thresh = np.median(a)
    atyp_lle = l[a >= thresh]
    typ_lle = l[a < thresh]
    if len(atyp_lle) < 15 or len(typ_lle) < 15:
        return None
    return float(atyp_lle.mean() - typ_lle.mean())


def main():
    print(f"=== ELBO-based surrogate-data test (AAFT, n={N_SURROGATES}), S&P 500 only ===\n")
    t0 = time.time()
    price_df = download_max("^GSPC", 20)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))

    real_gap = elbo_atypicality_and_lle(log_ret)
    print(f"real ELBO-based atypical-minus-typical LLE gap = {real_gap:+.4f}  ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(0)
    surrogate_gaps = []
    for s in range(N_SURROGATES):
        t1 = time.time()
        surr_ret = aaft_surrogate(log_ret, rng)
        g = elbo_atypicality_and_lle(surr_ret)
        if g is not None:
            surrogate_gaps.append(g)
        print(f"  surrogate {s+1}/{N_SURROGATES}: gap={g}  ({time.time()-t1:.0f}s)")
    surrogate_gaps = np.array(surrogate_gaps)

    rank_p = float(((surrogate_gaps <= real_gap).sum() + 1) / (len(surrogate_gaps) + 1))
    print(f"\nsurrogate null (n={len(surrogate_gaps)}): mean={surrogate_gaps.mean():+.4f} "
          f"std={surrogate_gaps.std():.4f}  min={surrogate_gaps.min():+.4f}  max={surrogate_gaps.max():+.4f}")
    print(f"rank-based one-sided p-value (real vs. surrogate null) = {rank_p:.4f}")


if __name__ == "__main__":
    main()
