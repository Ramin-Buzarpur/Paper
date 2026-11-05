# Paper framing notes — target: International Journal of Forecasting

Working title:
**"Regime-Dependent Predictability in Financial Markets: A Dynamical-Systems Account of
Novelty-Detection-Conditioned Forecasts"**

(Alternative, shorter: "Local Stability and Conditional Forecastability Across Asset Classes")

## Positioning (what changed vs. the original diffusion-model framing)

The paper is NOT "a diffusion model for financial forecasting." It is a **forecast-evaluation /
conditional-predictability** paper in the tradition of Diebold & Kilian's conditional
predictability measure and the regime-dependent predictability literature (e.g. bull/bear
regime work already published in this journal). The diffusion model is one of several
interchangeable novelty detectors (finding 33); the contribution is the *phenomenon* and its
*mechanism*, not the detector architecture.

## Abstract (draft)

> We study whether short-horizon financial return predictability is systematically conditional
> on a market's local state, rather than uniform across time. Using a delay-embedded
> phase-space representation of log-returns and a causal historical-analog forecasting method,
> we show across seven asset classes (equity index, two precious metals, two energy
> benchmarks, and two cryptocurrencies) that forecast accuracy is substantially higher on days
> flagged as locally atypical by an unsupervised novelty detector than on typical days — a
> result that is robust to the specific detector used (a diffusion-model ELBO score, Isolation
> Forest, Local Outlier Factor, and Mahalanobis distance in PCA space all produce the same
> qualitative pattern) and holds under both binary directional accuracy and a continuous,
> scale-free error metric (Mean Absolute Scaled Error against a random-walk benchmark). We then
> provide a first-principles dynamical explanation for this regime-dependence: using a
> Rosenstein-style estimator of the local (finite-time) Lyapunov exponent in the embedded phase
> space, we show that atypical days correspond to measurably more contracting local dynamics
> than typical days, across all seven assets (p < 0.001 in every case) — connecting the
> empirical predictability gap to the control-theoretic notion of local stability. A
> Dempster-Shafer fusion of the four novelty detectors further improves worst-case robustness,
> notably recovering a statistically significant effect on the S&P 500, the single asset where
> individual detectors were weakest. We extend the core finding into two additional,
> economically interpretable applications: a cross-asset composite atypicality index that
> behaves as a systemic-risk early-warning indicator around the 2008 and 2020 crises (while
> honestly under-responding to the narrower, sector-specific 2023 US regional-banking stress),
> and a directed Granger-causality network showing that equity-market atypicality leads
> commodity atypicality while cryptocurrency atypicality is dynamically segmented from both.

## Contribution statement (for intro / highlights)

1. A robust, cross-asset empirical demonstration of regime-dependent predictability, replicated
   across 4 independent unsupervised novelty-detection methods and validated with nested
   walk-forward evaluation, combinatorial purged cross-validation, and permutation testing.
2. A **dynamical-systems mechanism** (local Lyapunov exponents) explaining *why* atypical
   regimes are more predictable — not merely reporting *that* they are.
3. A **continuous, forecasting-standard evaluation** (MASE vs. random-walk, vs. classical AR(1))
   in addition to directional accuracy, addressing the standard "accuracy vs. a coin flip"
   objection.
4. Two generalizations beyond single-asset direction forecasting: a systemic-risk early-warning
   index and a directed cross-asset causality network — showing the phenomenon is a market
   stylized fact, not an artifact of one trading rule.
5. An honest reporting of boundary conditions and negative/null results throughout (S&P 500's
   weaker response; market breadth not improving index forecasts; the 2023 banking-crisis
   under-response) — consistent with the journal's emphasis on rigorous, non-overclaiming
   empirical forecast evaluation.

## Suggested classical benchmarks to report (beyond random-walk)
- AR(1) (causal, refit periodically) — see `arima_baseline_comparison.py` / Finding 43.
- (Optional, if time allows) a simple GARCH(1,1) volatility-only benchmark, framed as "does the
  edge survive once you control for volatility clustering" — not yet implemented; flag as future
  work if not done before submission.

## Author framing
Two authors with complementary technical backgrounds (computer engineering — forecasting
pipeline, validation methodology, software; electrical/control engineering — the dynamical-
systems/local-stability analysis) is a genuine, defensible division of contribution — state it
plainly in the author-contribution statement rather than implying a larger team.
