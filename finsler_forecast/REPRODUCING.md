# Reproducing the findings in README.md

This file exists so a reviewer (or anyone else) can independently re-run the headline results
in `README.md` without having to reverse-engineer the order the 46+ scripts were written in.
Everything here uses live Yahoo Finance data (`yfinance`), so exact numbers can drift by a
small amount run-to-run as history extends and as the diffusion net's random initialization
varies — the qualitative conclusions (direction, significance) have been checked to be stable
to that (see Finding 47's explicit parameter-sensitivity check).

## Environment

- Python: 3.11+ (developed and last verified on 3.14.6)
- Install: `pip install -r requirements.txt` from this directory. The committed
  `requirements.txt` additionally needs two packages used only by the classical-econometrics
  baselines, now added: `statsmodels>=0.14` (AR(1)/ARIMA, Granger causality) and `arch>=6.0`
  (GARCH(1,1)).
- A GPU is not required — the diffusion "ScoreNet" is a 2-layer MLP (`hidden=32`), not a large
  image-diffusion architecture; every script in this repo trains and runs in seconds-to-minutes
  per asset on CPU. `torch.cuda.is_available()` is used opportunistically if present.
- Internet access is required (all data is pulled live from Yahoo Finance at run time; nothing
  is vendored or cached in the repo).

## Core pipeline, in the order the headline claims depend on each other

Run each with `python <script>.py` from this directory. Each prints its own results to stdout;
nothing needs to be piped into anything else — every script re-downloads and re-derives what it
needs from raw prices.

1. **`economic_backtest.py`** — defines the shared building blocks (`embed`, `select_analogs`,
   the `ScoreNet` diffusion architecture, `train_scorenet`, `full_elbo_score`, `download_max`)
   that every later script imports. Running it directly reproduces the original economic
   backtest (Sharpe/drawdown) findings.
2. **`robustness_journal_grade.py`** — the core claim (Finding 33): diffusion ELBO vs. three
   classical anomaly detectors, plus the single-shot held-out confirmation. ~7 assets x
   ~4 minutes each.
3. **`statistical_rigor.py`** — the permutation-test machinery (`permutation_test`) reused by
   most later scripts; running it directly reproduces Finding 31.
4. **`local_lyapunov_regime.py`** — the dynamical-systems mechanism (Finding 41), full sample.
5. **`holdout_lyapunov_check.py`** — the held-out confirmation of the mechanism (Finding 46).
   Depends on `robustness_journal_grade.collect_full` and `local_lyapunov_regime`'s LLE helpers.
6. **`lyapunov_param_robustness.py`** — the Theiler-window / horizon sensitivity check
   (Finding 47). Depends on the same two modules as step 5.
7. **`continuous_error_metrics.py`**, **`arima_baseline_comparison.py`**,
   **`garch_baseline_comparison.py`** — the three continuous-error/classical-baseline
   comparisons (Findings 42-44), each with the paired-bootstrap significance test
   (`bootstrap_compare`, defined once in `continuous_error_metrics.py` and imported by the
   other two).
8. **`dempster_shafer_fusion.py`** — the Dempster-Shafer fusion of all 4 detectors
   (Finding 35). Depends on `robustness_journal_grade.collect_full`.
9. **`cross_asset_contagion.py`** — cross-asset co-occurrence (Finding 36). Independent of the
   above (recomputes its own fold data).
10. **`systemic_risk_indicator.py`** then **`granger_and_fresh_crisis.py`** — the composite
    systemic-risk index and Granger-causality network (Findings 38-40). The second script
    imports `causal_rank_series`/`ASSETS` from the first.
11. **`audit_leakage_check.py`** — the integrity audit (Finding 45): a side-by-side comparison
    of the pre-fix (self-inclusive) and post-fix (strictly causal) classical-detector scoring,
    on Brent and S&P 500 only (the spot-check that motivated fixing `robustness_journal_grade.py`
    and `cross_asset_contagion.py`'s detector-fitting window).

Every other `*.py` file in this directory not listed above (`vae_similarity_metric.py`,
`transformer_mae_metric.py`, `kan_similarity_metric.py`, `fractional_differentiation_metric.py`,
`derivative_jet_metric.py`, the various `*_gate.py` files, etc.) corresponds to one of the
earlier, individually-numbered findings (1-30) documented in `README.md` — each is
self-contained and runnable the same way (`python <script>.py`), independent of the above list.

## A note on what "reproducing" means here

Because every script re-downloads live Yahoo Finance history at run time, a run several months
from now will use a longer price history than the run that produced the numbers in `README.md`
— exact percentages and p-values will shift slightly. What should NOT change, and is what a
reviewer should check for, is: (a) the direction of every headline effect, and (b) whether the
7-asset panel remains predominantly significant. Finding 47's parameter-sensitivity check and
the two true single-shot holdout tests (Findings 33-B and 46) were run once, at the dates noted
in the README, and are reported as point-in-time results, not as a live, continuously-updating
claim.

## Known, honestly-disclosed limitations (see Finding 45 for the full audit)

- `statistical_rigor.permutation_test` reports a one-sided p-value (consistent with this
  investigation's single, pre-specified directional hypothesis) and has no `+1`
  numerator/denominator correction, so it can print `p=0.0000` where the more defensible
  statement is `p < 0.0005` (the 2000-shuffle test's resolution floor).
- Findings 35, 36, 38-40, and 43-44 (fusion, cross-asset contagion, the systemic-risk index,
  the Granger network, and the classical-baseline comparisons) are exploratory: tested on the
  full/overlapping sample, not on a genuinely untouched holdout, unlike Findings 33 and 46.
  They should be presented as suggestive/exploratory in any write-up, not as confirmed with the
  same strength as the two holdout-tested findings.
- Given how many model/architecture variants were tried over the course of this investigation
  before converging on the findings kept in this README, the overall process carries real
  "garden of forking paths" exposure (Gelman & Loken, 2013). The two headline claims (33, 46)
  address this directly via a true single-shot holdout rather than a post-hoc multiple-testing
  correction across every variant tried; the exploratory findings do not have this protection
  and are labeled accordingly above.
