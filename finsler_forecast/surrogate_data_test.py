"""
The single most field-standard validation missing from the local-Lyapunov mechanism (findings
41/46/47/48): a surrogate-data test (Theiler et al., 1992 -- the same Theiler whose exclusion
window this investigation already uses throughout). This is THE established method in
nonlinear time-series analysis for answering the question a dynamical-systems-literate reviewer
asks first: "is the atypical-vs-typical local-Lyapunov gap genuine nonlinear structure, or
could any series with the same linear autocorrelation and amplitude distribution as these
returns produce the same pattern by chance?"

Method: generate AAFT (Amplitude-Adjusted Fourier Transform) surrogates of each asset's
log-return series. AAFT surrogates preserve the amplitude distribution AND the linear
autocorrelation/power spectrum of the original series exactly, while destroying any nonlinear
temporal structure beyond what a linear Gaussian process with that spectrum would produce. If
the atypical-vs-typical LLE gap is genuine nonlinear structure, it should be far larger in the
real data than in surrogates that share only the real data's LINEAR properties.

To make 200 surrogates per asset computationally tractable, atypicality here is defined by a
cheap, model-free causal k-NN distance score (not the diffusion net) -- consistent with
finding 33's conclusion that the phenomenon is detector-agnostic, so this substitution does not
change what is being tested, only how expensively "atypical" is computed per surrogate.

Result (see README Finding 49): printed output / README has the actual numbers.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from economic_backtest import embed, download_max, HORIZON, M, TAU, THEILER

ASSETS = [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20)]
N_SURROGATES = 200


def aaft_surrogate(x, rng):
    """Amplitude-Adjusted Fourier Transform surrogate (Theiler et al., 1992): preserves the
    amplitude distribution and (approximately) the power spectrum of x, randomizes phases."""
    n = len(x)
    gauss = np.sort(rng.normal(size=n))
    rank_x = np.argsort(np.argsort(x))
    matched = gauss[rank_x]  # same rank order as x, Gaussian amplitudes

    spec = np.fft.rfft(matched)
    phases = rng.uniform(0, 2 * np.pi, size=len(spec))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    randomized = np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=n)

    rank_r = np.argsort(np.argsort(randomized))
    sorted_x = np.sort(x)
    return sorted_x[rank_r]


def knn_atypicality_and_lle(pc, n_pts, origins, k=10, horizon=HORIZON, theiler=THEILER):
    """Cheap, model-free causal atypicality (distance to k-th causal nearest neighbor) and
    local Lyapunov exponent, for one series (real or surrogate)."""
    atyp, lle = {}, {}
    for j in origins:
        hist = pc[:j]
        if len(hist) <= theiler + k:
            continue
        d = np.sum((hist - pc[j]) ** 2, axis=1)
        order = np.argsort(d)
        causal_order = [int(c) for c in order if j - int(c) > theiler]
        if len(causal_order) < k:
            continue
        atyp[j] = float(np.sqrt(d[causal_order[k - 1]]))

        nn = causal_order[0]
        j_f, nn_f = j + horizon, nn + horizon
        if j_f >= n_pts or nn_f >= n_pts:
            continue
        d_now = np.linalg.norm(pc[j] - pc[nn])
        d_future = np.linalg.norm(pc[j_f] - pc[nn_f])
        if d_now < 1e-10:
            continue
        lle[j] = float(np.log(max(d_future, 1e-10) / d_now) / horizon)
    return atyp, lle


def atypical_minus_typical_gap(log_ret, min_history=250):
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    origins = list(range(min_history, n_pts - HORIZON - 1, HORIZON))
    atyp, lle = knn_atypicality_and_lle(pc, n_pts, origins)
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
    print(f"=== Surrogate-data test (AAFT, n={N_SURROGATES}) for the local-Lyapunov mechanism ===\n")
    for ticker, name, max_years in ASSETS:
        t0 = time.time()
        price_df = download_max(ticker, max_years)
        prices = price_df["Close"].to_numpy(dtype=np.float64)
        log_ret = np.diff(np.log(prices))

        real_gap = atypical_minus_typical_gap(log_ret)
        if real_gap is None:
            print(f"  {name}: skipped (too few usable folds)")
            continue

        rng = np.random.default_rng(0)
        surrogate_gaps = []
        for s in range(N_SURROGATES):
            surr_ret = aaft_surrogate(log_ret, rng)
            g = atypical_minus_typical_gap(surr_ret)
            if g is not None:
                surrogate_gaps.append(g)
        surrogate_gaps = np.array(surrogate_gaps)

        rank_p = float(((surrogate_gaps <= real_gap).sum() + 1) / (len(surrogate_gaps) + 1))
        print(f"{name} ({time.time()-t0:.0f}s, {len(surrogate_gaps)} usable surrogates):")
        print(f"  real atypical-minus-typical LLE gap = {real_gap:+.4f}")
        print(f"  surrogate null: mean={surrogate_gaps.mean():+.4f}  std={surrogate_gaps.std():.4f}  "
              f"min={surrogate_gaps.min():+.4f}  max={surrogate_gaps.max():+.4f}")
        print(f"  rank-based one-sided p-value (real gap vs. surrogate null) = {rank_p:.4f}\n")


if __name__ == "__main__":
    main()
