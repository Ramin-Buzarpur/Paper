"""
A genuine information-fusion contribution (vs. findings 31-33's side-by-side COMPARISON of
detectors): combine the 4 anomaly scores (diffusion ELBO, Isolation Forest, LOF,
Mahalanobis-PCA) into one belief via Dempster-Shafer (DS) evidence theory -- the classical
fusion framework used throughout the Information Fusion literature (Chen & Aickelin 2006;
numerous DST-ensemble anomaly-detection papers since).

Mechanism: each detector's causal percentile rank (0..1, "how atypical is today relative to
history so far") becomes a basic probability assignment with a small reserved uncertainty mass:
  m_i(Atypical) = rank_i * (1 - u),  m_i(Typical) = (1 - rank_i) * (1 - u),  m_i(Theta) = u
Combined pairwise via Dempster's rule of combination (renormalizing out the conflict mass K)
across all 4 detectors, giving one fused belief in "Atypical" per day. This is a real fusion
operation, not an average or a vote -- conflicting detectors are explicitly discounted via the
DST conflict term.

Result (see README Finding 35): the fused signal is the best entry on 2 of 7 assets and never
far from the best on the rest -- fusion improves worst-case robustness rather than chasing the
single best detector per asset:

| asset | best individual detector | DS-fused |
|---|---|---|
| Brent | 13.3pp (LOF) | **15.2pp** -- beats all 4 |
| S&P 500 | 5.3pp, p=0.014 (LOF) | **6.3pp, p=0.004** -- beats all 4, strongest S&P 500 result in the whole investigation |
| Gold | 10.4pp (Isolation Forest) | 8.8pp (close, not best) |
| Bitcoin | 13.4pp (Isolation Forest) | 12.4pp (close) |
| Ethereum | 13.8pp (Isolation Forest/Mahalanobis) | 12.7pp (close) |
| Silver | 13.8pp (Mahalanobis) | 13.5pp (near-tied) |
| WTI Crude | 10.2pp (diffusion) | 9.0pp (close) |

Fusion never badly underperforms the best individual detector, and on the two assets where a
single detector's choice mattered most (Brent, and especially S&P 500 -- the panel's weak
point throughout this whole investigation), fusion strictly wins. This is the standard
information-fusion thesis (diverse evidence sources combined via a principled rule improve
worst-case robustness over picking any single source) demonstrated empirically, not asserted.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from robustness_journal_grade import collect_full, ASSETS
from statistical_rigor import permutation_test


def ds_combine_two(bel_a, bel_u_a, bel_b, bel_u_b):
    """Dempster's rule of combination for a 2-element frame {Atypical, Typical} with an
    uncertainty (Theta) mass. bel_* are P(Atypical) masses, *_u are the Theta (uncertainty)
    masses; P(Typical) mass is (1 - bel - u)."""
    typ_a, typ_b = 1 - bel_a - bel_u_a, 1 - bel_b - bel_u_b
    # conflict: Atypical-vs-Typical disagreement mass
    k = bel_a * typ_b + typ_a * bel_b
    denom = max(1 - k, 1e-9)
    m_atyp = (bel_a * bel_b + bel_a * bel_u_b + bel_u_a * bel_b) / denom
    m_u = (bel_u_a * bel_u_b) / denom
    return m_atyp, m_u


def fuse_scores(df, cols, min_train=20, uncertainty=0.15):
    n = len(df)
    ranks = {}
    for col in cols:
        vals = df[col].to_numpy()
        r = np.zeros(n)
        for i in range(min_train, n):
            r[i] = (vals[:i] < vals[i]).mean()  # causal percentile rank
        ranks[col] = r

    fused = np.zeros(n)
    for i in range(n):
        m_atyp, m_u = ranks[cols[0]][i] * (1 - uncertainty), uncertainty
        for col in cols[1:]:
            m_atyp, m_u = ds_combine_two(m_atyp, m_u, ranks[col][i] * (1 - uncertainty), uncertainty)
        fused[i] = m_atyp
    return fused


COLS = ["diff_atyp", "iso", "lof", "maha"]

if __name__ == "__main__":
  for ticker, name, max_years in ASSETS:
    t0 = time.time()
    df = collect_full(ticker, name, max_years)
    if len(df) < 60:
        print(f"{name}: too few folds, skipping")
        continue
    df["fused"] = fuse_scores(df, COLS)
    print(f"\n=== {name} (n={len(df)}, {time.time()-t0:.0f}s) ===")
    for col, label in zip(COLS + ["fused"], ["diffusion", "Isolation Forest", "LOF", "Mahalanobis-PCA", "DS-FUSED"]):
        obs, p, _ = permutation_test(df, col)
        marker = " <-- fusion" if col == "fused" else ""
        print(f"  {label:18s} gap={obs*100:+5.1f}pp  p={p:.4f}{marker}")
