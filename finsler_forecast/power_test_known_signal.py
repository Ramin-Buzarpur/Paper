"""
Decisive power test: does the FIXED pipeline retain the ability to detect a genuine,
designed-in, atypicality-conditioned predictability signal -- or did the leakage fix also
break the pipeline's sensitivity (which would make the "nothing survives on real data" result
uninformative, per the type-2 concern)?

Method: generate a synthetic return series via a known nonlinear generative rule --
magnitude-conditional mean reversion: after an abnormally large |return| (exceeding a
threshold), the NEXT return is deliberately biased in the opposite direction with a known,
fixed strength; otherwise returns are pure IID noise. This is exactly the kind of
"locally-extreme-state implies more predictable near-term direction" structure this whole
investigation originally hypothesized might exist in real markets -- built here with a known
ground truth instead of hoped-for real structure.

If the fixed pipeline has power, it should detect a large, highly significant
atypicality-conditioned accuracy gap on this series (since the atypicality score, built from
the embedding's magnitude, should flag exactly the "after a big move" states where the
designed reversion signal is active).

Result (see README Finding 52): printed output / README has the actual numbers.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from surrogate_data_test_core_claim_elbo import collect_real_pipeline
from statistical_rigor import permutation_test
from economic_backtest import download_max


def generate_conditional_reversion_series(n, sigma, threshold_std=1.0, reversion_strength=0.6,
                                           bias_steps=3, seed=0):
    """r[t] = eps[t] normally; but if |r[t-1]| > threshold_std*sigma, the NEXT `bias_steps`
    returns (not just one) are ALL biased toward -sign(r[t-1]) with the given strength -- so the
    full HORIZON-step compounded future return (the sum the pipeline actually tests) carries an
    undiluted directional push, not just one of its three summed terms."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=n)
    r = eps.copy()
    t = 1
    while t < n:
        if abs(r[t - 1]) > threshold_std * sigma:
            bias = -reversion_strength * np.sign(r[t - 1]) * sigma
            for s in range(bias_steps):
                if t + s < n:
                    r[t + s] = eps[t + s] + bias
            t += bias_steps  # avoid re-triggering mid-biased-window on the biased values themselves
        else:
            t += 1
    return r


def main():
    print("=== Power test: can the FIXED pipeline detect a known, designed-in signal? ===\n")
    price_df = download_max("BZ=F", 20)
    prices_real = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret_real = np.diff(np.log(prices_real))
    sigma = float(np.std(log_ret_real))
    n = len(log_ret_real)

    for strength, label in [(0.0, "control: strength=0 (should behave like pure noise)"),
                             (0.6, "designed signal: strength=0.6 (should be clearly detected)")]:
        t0 = time.time()
        synth_ret = generate_conditional_reversion_series(n, sigma, reversion_strength=strength, seed=1)
        synth_prices = np.concatenate([[prices_real[0]], prices_real[0] * np.exp(np.cumsum(synth_ret))])
        df = collect_real_pipeline(synth_prices, synth_ret)
        if len(df) < 60:
            print(f"{label}: skipped (too few folds)")
            continue
        gap, p, _ = permutation_test(df, "atypicality")
        overall_acc = df["correct"].mean()
        print(f"{label}:")
        print(f"  n_folds={len(df)}  overall accuracy={overall_acc*100:.1f}%  "
              f"atypicality-conditioned gap={gap*100:+.2f}pp  p={p:.4f}  ({time.time()-t0:.0f}s)\n")


if __name__ == "__main__":
    main()
