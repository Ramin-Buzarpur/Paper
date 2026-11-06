<!-- Working draft. Target venue: International Journal of Forecasting (or a comparable
     forecasting-methodology journal). Structure follows IJF's guide for authors: short
     abstract (100-150 words), numbered sections, minimal notation, references by
     author-year. This file is a draft for internal iteration -- not a camera-ready
     manuscript. -->

# Spurious Predictability from an Embedding Index Error: A Case Study in Financial Analog
# Forecasting

**Authors:** [Author 1, Computer Engineering], [Author 2, Electrical/Control Engineering]

## Abstract (draft, ~150 words)

We document and trace to its root cause a case of spurious predictability in
delay-embedding-based financial forecasting. Testing four novelty/anomaly detectors across
seven asset classes, we initially found forecast accuracy substantially higher on atypical
days -- a result that survived walk-forward cross-validation, permutation testing, and a
genuine out-of-sample holdout. It was an artifact of a one-step index error: the embedding's
most recent component was numerically identical to the first step of the future return being
predicted. We confirm this diagnosis with an algebraic proof, a pipeline-free correlation test,
a null control on synthetic white noise, and a power test showing the corrected pipeline still
detects genuine designed-in signal; all four detectors were affected identically, pointing to
the shared embedding rather than any one method. We propose a null-noise-plus-surrogate-data
protocol that would have caught this error in hours rather than months, and recommend it as
standard practice before reporting a positive result in this class of study.

## 1. Introduction

Short-horizon financial return forecasting via historical-analog methods -- delay-embed recent
returns into a low-dimensional state space, locate the $k$ nearest historical analogs, and
average their realized continuations -- is a long-standing technique with roots in the Analog
Ensemble literature from atmospheric and space-weather forecasting (Delle Monache et al.,
2011) and in the broader nonlinear-dynamics tradition of state-space reconstruction (Takens,
1981; Kantz & Schreiber, 2004). A natural extension, explored across a large and active recent
literature, is to condition forecast confidence or forecast usage on an auxiliary novelty- or
anomaly-detection score computed on the same embedded state: the intuition is that "unusual"
market states might be more (or less) predictable than "typical" ones.

This paper began as an attempt to formalize and validate exactly that intuition. Using a
diffusion-model reconstruction score (and, for robustness, three classical novelty detectors:
Isolation Forest, Local Outlier Factor, and Mahalanobis distance in PCA space) as the
atypicality signal, we found -- across seven asset classes (equity index, two energy
benchmarks, two precious metals, two cryptocurrencies) -- that forecast accuracy conditional
on high atypicality was substantially and significantly higher than conditional on low
atypicality. The result replicated under nested walk-forward evaluation, combinatorial purged
cross-validation, formal permutation testing (Good, 2005), a genuine single-shot 20% holdout
that was never touched during model development, and even a proposed dynamical-systems
explanation involving local Lyapunov exponents.

It was wrong. Section 4 describes how we found out, and why none of the validation steps
listed above caught it. We believe the diagnostic protocol we used to find and confirm the
error -- and which we recommend other researchers adopt -- is the paper's main contribution.

### 1.1 Contributions

1. We document a previously-unreported class of index error in delay-embedding-based
   forecasting pipelines, under which the embedding's most recent feature is numerically
   identical to part of the forecast target, and show it survives methods normally considered
   strong safeguards against overfitting (walk-forward CV, permutation testing, genuine
   out-of-sample holdout).
2. We verify the diagnosis at six increasing levels of pipeline complexity, from a closed-form
   algebraic proof to a null-hypothesis control on synthetic data with a provably known
   absence of structure -- noting explicitly that this is six checks at different levels of
   abstraction, not six independent sources of evidence (checks 1 and 3 are two
   representations of the same underlying fact).
3. We show the error affected all four structurally different detectors tested identically,
   establishing that the apparent "finding" was a property of the shared embedding
   convention, not of any one detection method.
4. We propose a lightweight, two-part diagnostic protocol (a pure-noise null control, and
   AAFT surrogate-data testing) that would have caught this error immediately, and that we
   recommend as a standard pre-submission check for this entire class of method.

## 2. Related Work

**Delay embedding and analog forecasting.** The analog-forecasting technique underlying this
investigation -- locate the $k$ nearest historical states to the current embedded state and
pool their verifying continuations -- is structurally identical to the Analog Ensemble (AnEn)
framework developed in the atmospheric-forecasting literature. Hamill & Whitaker (2006)
introduced the underlying analog-correction technique for precipitation reforecasts; Delle
Monache et al. (2011) extended it with a Kalman-filter/analog post-processing scheme, and Delle
Monache et al. (2013) formalized the probabilistic AnEn method as a way of generating an
ensemble forecast directly from a single deterministic trajectory. Alessandrini et al. (2015)
demonstrate AnEn's practical value for short-term wind-power forecasting, illustrating the
framework's reach beyond its original precipitation application. The state-space
reconstruction underlying the embedding itself follows the standard delay-coordinate approach
of Takens (1981) and the broader nonlinear time-series methodology surveyed in Kantz &
Schreiber (2004). To our knowledge, the present investigation (and the broader project it is
drawn from) is the first to apply the AnEn framework, under that name, to financial return
series rather than to a meteorological or physical system.

**Spurious predictability and leakage in financial machine learning.** The closest precedents
to this paper's diagnostic contribution are Nikolopoulos (2026), *"Spurious Predictability in
Financial Machine Learning"* (arXiv:2604.15531), which formalizes a falsification framework
for diagnosing exactly this class of problem in financial ML pipelines; the foundational survey
on leakage formulation, detection, and avoidance in data mining more broadly (Kaufman, Rosset,
Perlich & Stitelman, 2012, *ACM Transactions on Knowledge Discovery from Data*); and a 2025/2026
preprint demonstrating that decomposition-ensemble cryptocurrency forecasting models' reported
directional accuracy is attributable almost entirely to look-ahead bias inherent in two-sided
signal-decomposition methods (CEEMDAN/VMD), which proposes a falsification test structurally
similar in spirit to the null-noise control used here (Research Square preprint, DOI:
10.21203/rs.3.rs-11529129/v1; author to be confirmed before final submission). On the
statistical side, the multiplicity concerns raised by testing many model/detector variants on
the same panel are formalized by White's (2000) Reality Check for data snooping and by the
"garden of forking paths" framework of Gelman & Loken (2013/2014) and its direct application to
financial-anomaly research by Coqueret (2024).

**Surrogate-data testing and the chaos-in-markets debate.** Our diagnostic protocol's second
component draws directly on the surrogate-data methodology introduced by Theiler, Galdrikian,
Longtin, Eubank & Farmer (1992) for testing nonlinearity against a matched-linear null, and
refined by Schreiber & Schmitz's (1996, 2000) Iterative Amplitude-Adjusted Fourier Transform
(IAAFT), a closer-to-exact refinement of the AAFT surrogates used in this paper. This
methodology underlies the decades-long, still-unsettled literature testing whether financial
return series exhibit genuine deterministic chaos rather than linear stochastic structure that
merely resembles it (surveyed critically in Barnett & Serletis, 2000, *Journal of Economic
Dynamics and Control*); studies applying surrogate-data and related nonlinearity tests to
individual equity indices have reached conflicting conclusions depending on denoising and test
choice, underscoring that this is an active methodological debate rather than a settled
question. This literature's central lesson -- that apparent nonlinear structure in a detector
statistic must be checked against a matched linear surrogate null before being attributed to
genuine dynamics -- is the lesson this paper's own investigation initially failed to apply to
its *own* core predictability claim, not only to its exploratory dynamical-systems
explanation.

## 3. Data and Baseline Method

### 3.1 Data
Daily close prices for seven assets, downloaded from Yahoo Finance: Brent crude (`BZ=F`), the
S&P 500 index (`^GSPC`), gold (`GC=F`), silver (`SI=F`), WTI crude (`CL=F`), Bitcoin
(`BTC-USD`), and Ethereum (`ETH-USD`), using the maximum available history per asset (8-20
years depending on listing date).

### 3.2 Delay embedding and analog selection
Log returns $r_t = \log(P_t / P_{t-1})$ are delay-embedded into $\mathbb{R}^M$ with $M=3$,
$\tau=1$: $x_j = (r_{t}, r_{t-\tau}, r_{t-2\tau})$. For a forecast origin, the $K=7$ nearest
historical analogs (by a learned feature distance; see 3.3) are selected causally, excluding a
Theiler window of 10 days around the origin to avoid trivial temporal auto-matches (Theiler,
1986). The forecast is the sign of the mean $H=3$-day forward return realized by the selected
analogs.

### 3.3 Atypicality detectors
Four detectors score each origin's "atypicality": (i) the reconstruction loss (ELBO) of a
small denoising diffusion model (a 2-layer MLP, hidden width 32) trained causally on the
embedded history; (ii) Isolation Forest; (iii) Local Outlier Factor; (iv) Mahalanobis distance
in a PCA-reduced space with Ledoit-Wolf shrinkage. All four are fit causally, retrained
periodically on history up to the current origin only.

### 3.4 Evaluation
Nested nonoverlapping walk-forward folds (step size $H$); a 2000-resample permutation test for
the accuracy-gap statistic (median-split by atypicality score); Fisher's method to combine
p-values across the seven-asset panel; and, separately, a single untouched 20% holdout per
asset, evaluated exactly once.

## 4. The Error: Discovery, Diagnosis, and Fix

### 4.1 The original result
Using the method of Section 3, we found a substantial, statistically significant gap in
forecast accuracy between high- and low-atypicality origins: for example, the diffusion-ELBO
detector showed an +8.3 percentage-point accuracy gap on Brent crude ($p<0.0001$), and three
classical detectors (Isolation Forest, LOF, Mahalanobis-PCA) showed comparable or larger gaps
on the same and other assets (Table 1 reports the full pre-fix comparison across four
detectors and seven assets). A genuine single-shot holdout test, touched for the first time
after all model development was frozen, reproduced the effect at the panel level
(Fisher-combined $p = 3.8\times 10^{-5}$ across seven assets). We additionally proposed and
validated a dynamical-systems explanation (local Lyapunov exponents), which itself replicated
on a second, independent holdout and was robust to parameter choices and to splitting the
sample by calendar era.

### 4.2 The surrogate-data control that exposed the problem
Motivated by a standard robustness check for the dynamical-systems explanation specifically
(not yet the core claim), we generated AAFT surrogate return series -- matching the real data's
amplitude distribution and linear power spectrum exactly, with phases randomized to destroy
any nonlinear temporal structure -- and re-ran the local-Lyapunov analysis. The purely linear
surrogates reproduced (and in fact exceeded) the real data's effect size, for both a cheap
$k$-NN atypicality proxy and the production diffusion-ELBO detector. This result, on its own,
is explicable without positing any coding error (surrogate-data literature is full of
"apparent nonlinear structure is actually linear" findings) -- but it prompted us to apply the
same control to the paper's actual central claim: the atypicality-conditioned *accuracy* gap,
not merely the auxiliary Lyapunov statistic.

### 4.3 The decisive control: pure white noise
We generated literal i.i.d. Gaussian white noise -- a process with, by construction, zero
genuine forecastable structure of any kind -- and ran the identical pipeline on it. The
pipeline reported a "significant" atypicality-conditioned accuracy gap of 5-7 percentage
points, with permutation $p$-values as low as the resolution floor of a 2000-resample test
(0.0000), on every one of three initial noise draws (a result extended to ten trials across
two assets in Section 5, with the same conclusion). A process with zero genuine predictive
structure, by construction, cannot produce a genuine 5-7 point accuracy edge; this result can
only be explained by a defect in the measurement procedure itself.

### 4.4 Root cause
In plain terms: the pipeline's "most recent known return" -- the feature used to judge how
unusual today looks and which historical days resemble it -- was not actually known yet at the
moment the forecast was made. It was one day into the future the model was supposedly
predicting.

More precisely: with origin index $j$, embedding length $M$, and delay $\tau$, the project's
convention located "today's price" at array index $p = j + (M-1)\tau$ and defined the forecast
target as the $H$-step return beginning at $p$. But the embedding's most recent component,
$x_j^{(0)}$, is constructed from $\texttt{log\_ret}[j + (M-1)\tau]$ -- exactly
$\texttt{log\_ret}[p]$: the **first of the $H$ returns summed to produce the forecast target
itself**. The correct convention is $p = j + M\tau$, which places the embedding entirely in the
past relative to the forecast window, with zero gap and zero overlap.

### 4.5 Why none of the standard safeguards caught it
The error sits in every fold's feature-construction code identically, in both the "training"
and the "held-out" portions of the data -- it is a fixed, deterministic property of the index
arithmetic, not something learned from or overfit to any particular sample. This is precisely
why each of the three safeguards this literature relies on most heavily missed it:

- **Walk-forward cross-validation** protects against overfitting to one historical window. It
  offers no protection against a structural bias present identically in every window.
- **A genuine single-shot holdout** protects against re-using information learned during model
  selection. It offers no protection against a bias baked into the feature-construction code
  itself, which is applied, by construction, the same way to the holdout period as to
  everything else.
- **Permutation testing** protects against mistaking sampling noise for a real effect. It
  offers no protection when the "effect" is a deterministic artifact that reappears, with the
  same mechanism intact, in every permutation.

Each of these three tools is, individually, considered strong evidence in the forecasting
literature -- and none of them is designed to catch a fixed arithmetic bias in feature
construction, because none of them varies the feature-construction code itself. A fourth tool
does: comparing the pipeline's output on real data against its output on data engineered to
contain no real structure at all (Section 4.3; formalized as a general protocol in Section
7.2) directly tests the feature-construction code, not the sample.

## 5. Verification

Because of the scale of what the diagnosis implies -- the central claim of a multi-month
investigation turning out to be an artifact -- we did not accept it on the strength of any
single check. The six checks below span increasing levels of pipeline complexity, from a
closed-form algebraic identity to the full production pipeline; checks 1 and 3 are two
representations of the same underlying fact rather than independent evidence, and are listed
separately only because they operate at different levels of abstraction (full detail and
numbers: README Findings 51-52 of the accompanying code repository).

1. **Exact algebraic identity**: hand-derived and numerically confirmed that $x_j^{(0)}$ equals
   $\texttt{log\_ret}[p]$ exactly, not approximately, on a tiny synthetic price array.
2. **Generalization across the real series**: confirmed over 33 (resp. 47, in a fully
   independent re-implementation) origins spread across real Brent history: the overlap holds
   universally under the old formula and never under the corrected one.
3. **A pipeline-free, model-free direct test**: $\text{sign}(x_j^{(0)})$ matches
   $\text{sign}(\text{actual future return})$ in 68.3% of folds under the old convention versus
   50.1% (chance) under the fix -- 68.3% rather than $\approx$100% is itself consistent with the
   diagnosis, since the leaked component is only one of the three summands making up the
   target.
4. **A null control repeated ten times** across two assets and two random seeds: under the
   fixed pipeline, mean permutation $p$-value across trials was 0.496 (the theoretical
   expectation under a true null is exactly 0.5), with a false-positive rate matching the
   nominal 5% level.
5. **A fully independent, from-scratch script with zero imports from the project's own
   codebase**, using a fresh data download and hand-written embedding/index arithmetic,
   reproduced the identical result, ruling out a shared misreading of the code baked into the
   investigation's own tooling.
6. **A power test with a known, designed-in signal**: a synthetic series with an explicit,
   by-construction magnitude-conditional mean-reversion rule (after any return exceeding one
   standard deviation, the next three returns are deliberately biased in the opposite
   direction) was correctly detected by the fixed pipeline (accuracy 52.2%$\to$58.2%, gap
   $p=0.011$), confirming the fix did not simply desensitize the method.

We additionally ran a systematic six-point leakage audit beyond the specific bug (data-feature
temporal ordering; embedding-normalization causality; train/test shuffling; target-feature
alignment; residual embedding overlap; detector self-inclusion). One further, independent, and
much smaller issue was found -- the embedding's scale normalization is computed over the full
series rather than causally -- disclosed, and shown empirically not to materially affect the
result (Brent: $+2.4$pp $\to -0.1$pp under a corrected causal-scale variant, both
non-significant; S&P 500: $-1.2$pp $\to -1.2$pp, materially unchanged).

## 6. Results After Correction

**Table 2. Diffusion-ELBO accuracy gap, corrected pipeline, by asset.**

| Asset | Accuracy gap | $p$-value |
|---|---|---|
| Brent | +2.4pp | 0.176 |
| S&P 500 | -1.2pp | 0.692 |
| Gold | -5.4pp | 0.985 |
| Bitcoin | -1.7pp | 0.720 |
| Ethereum | +1.2pp | 0.352 |

No accuracy gap reaches significance; several point estimates are negative. Extending the
comparison to all four detectors across the full seven-asset panel (28 tests; see README
Finding 53 for the complete table) finds exactly 1 of 28 cells below $p=0.05$
(Ethereum/Mahalanobis-PCA, $p=0.025$) -- indistinguishable from the $\approx$1.4 false
positives expected by chance alone at $\alpha=0.05$ across 28 independent tests.

The genuine single-shot holdout test, re-run with the corrected pipeline, tells the same
story at the panel level: every asset is individually non-significant, and Fisher's combined
$p$-value across the seven-asset panel is **0.288**, against the original (uncorrected)
pipeline's $p = 3.8\times 10^{-5}$ on the identical protocol -- the strongest-looking evidence
in the original investigation collapses to a result indistinguishable from chance.

## 7. Discussion

### 7.1 Why this is worth publishing
A null result reached through careful, multiply-verified diagnosis of a subtle and previously
undocumented methodological error is, we argue, more valuable to the forecasting community
than the original (incorrect) positive finding would have been, for three reasons. First, the
error is structural to a widely-used class of method (delay-embedding plus auxiliary
novelty/anomaly scoring), not specific to our implementation choices, and is therefore likely
to recur in other studies using the same convention. Second, we show concretely that the
standard toolkit of safeguards against overfitting in this literature -- walk-forward
validation, permutation testing, even genuine out-of-sample holdout testing -- did not detect
it, which is itself an actionable finding about the limits of those safeguards for this
specific failure mode. Third, we provide a lightweight, generally-applicable diagnostic (a
null-noise control plus surrogate-data testing) that would catch this class of error cheaply,
before a researcher invests months in building an elaborate positive-finding narrative on top
of it, as we did.

### 7.2 A recommended diagnostic protocol
We recommend that any study using delay-embedding plus an auxiliary detection score for
financial (or other) time-series forecasting run, prior to reporting a positive result, at
minimum: (i) a null-data control -- apply the identical pipeline to synthetic data with
provably zero real structure (i.i.d. noise matching the real series' scale is sufficient) and
confirm the reported effect does not appear; (ii) a surrogate-data control -- apply the
identical pipeline to AAFT (or comparable) surrogates of the real series, to separate genuine
structure from an artifact reproducible by matched linear noise. Both checks are inexpensive
relative to the modeling effort typically invested in this class of study, and both would have
caught the error documented in this paper within hours rather than months.

### 7.3 Limitations
This paper documents one specific error, in one specific project's specific implementation of
a standard method family. We do not claim the error is present in any other published study;
we claim only that the general class of convention (using the embedding's most-recent-return
component without an explicit, verified alignment check against the forecast target's own
starting index) carries this risk, and that the diagnostic protocol we propose is a cheap,
general-purpose safeguard regardless of whether any specific paper happens to contain this
exact error.

## 8. Conclusion

We set out to validate a claim that market states flagged as locally atypical by an
unsupervised novelty detector are more forecastable than typical states. The claim initially
passed every standard validation step used in this literature. It did not survive a null-data
control, because the apparent effect was in fact a one-step index error that gave the
forecasting model partial access to its own target. We confirm this diagnosis at six
increasing levels of pipeline complexity and show the corrected pipeline retains full power to
detect genuine designed-in signal, ruling out that the correction merely broke the method. We
offer the diagnostic protocol that exposed the error -- a null-noise control and a
surrogate-data test, both standard tools in adjacent literatures but, to our knowledge, not
yet standard practice in this specific subfield -- as this paper's primary, generalizable
contribution.

## References

Alessandrini, S., Delle Monache, L., Sperati, S., & Nissen, J. (2015). A novel application of
an analog ensemble for short-term wind power forecasting. *Renewable Energy*, 76, 768-781.
https://doi.org/10.1016/j.renene.2014.11.061

Barnett, W. A., & Serletis, A. (2000). Martingales, nonlinearity, and chaos. *Journal of
Economic Dynamics and Control*, 24(5-7), 703-724.

Coqueret, G. (2024). Forking paths in financial economics. *arXiv preprint* arXiv:2401.08606.

Delle Monache, L., Nipen, T., Liu, Y., Roux, G., & Stull, R. (2011). Kalman filter and analog
schemes to post-process numerical weather predictions. *Monthly Weather Review*, 139,
3554-3570.

Delle Monache, L., Eckel, F. A., Rife, D. L., Nagarajan, B., & Searight, K. (2013).
Probabilistic weather prediction with an analog ensemble. *Monthly Weather Review*, 141,
3498-3516. https://doi.org/10.1175/MWR-D-12-00281.1

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business &
Economic Statistics*, 13(3), 253-263.

Gelman, A., & Loken, E. (2013). The garden of forking paths: Why multiple comparisons can be a
problem, even when there is no "fishing expedition" or "p-hacking" and the research hypothesis
was posited ahead of time. Department of Statistics, Columbia University (working paper);
published in revised form as "The Statistical Crisis in Science," *American Scientist*, 102(6),
460 (2014).

Good, P. I. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.).
Springer Series in Statistics. Springer-Verlag.

Hamill, T. M., & Whitaker, J. S. (2006). Probabilistic quantitative precipitation forecasts
based on reforecast analogs: Theory and application. *Monthly Weather Review*, 134(11),
3209-3229. https://doi.org/10.1175/MWR3237.1

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy.
*International Journal of Forecasting*, 22(4), 679-688.

Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis* (2nd ed.). Cambridge
University Press.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining:
Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*,
6(4), Article 15. https://doi.org/10.1145/2382577.2382579

Nikolopoulos, S. D. (2026). Spurious predictability in financial machine learning. *arXiv
preprint* arXiv:2604.15531.

[Research Square preprint] (2025/2026). The leakage premium in decomposition-based
cryptocurrency forecasting. DOI: 10.21203/rs.3.rs-11529129/v1. *(Author name to be confirmed
from the published record before final submission -- not independently verifiable at the time
of this draft.)*

Schreiber, T., & Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. *Physical
Review Letters*, 77, 635-638.

Schreiber, T., & Schmitz, A. (2000). Surrogate time series. *Physica D*, 142, 346-382.

Takens, F. (1981). Detecting strange attractors in turbulence. In *Dynamical Systems and
Turbulence, Warwick 1980* (pp. 366-381). Springer.

Theiler, J. (1986). Spurious dimension from correlation algorithms applied to limited
time-series data. *Physical Review A*, 34(3), 2427.

Theiler, J., Galdrikian, B., Longtin, A., Eubank, S., & Farmer, J. D. (1992). Testing for
nonlinearity in time series: The method of surrogate data. *Physica D*, 58, 77-94.

White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.

---
*Internal note: a full numeric audit (Abstract vs. Results tables vs. README.md) was run on
2026-11-06 and found no discrepancies: the Fisher-combined p-values (0.000038 pre-fix, 0.288
post-fix), the 1-of-28 detector-comparison result, the 68.3%/50.1% sign-correlation figures,
and the normalization-check numbers (Brent +2.4pp -> -0.11pp) all match between this draft and
the source-of-truth repository. The "six independent ways" phrasing was corrected throughout to
"six checks at different levels of pipeline complexity" with an explicit note that checks 1 and
3 are two representations of one fact, not independent evidence. References are now complete
and verified against primary sources, except the decomposition-leakage cryptocurrency
preprint's author, which could not be independently confirmed and is flagged for manual
verification before submission.

A full prose pass was completed on 2026-11-06, in the order: Abstract (trimmed from ~190 to 156
words to fit IJF's 100-150-word guideline, with minor tolerance), Section 4 "The Error" (added
concrete example numbers in 4.1 instead of a bare table pointer; added a plain-language framing
sentence to 4.4 before the formal derivation; restructured 4.5 into bulleted per-safeguard
explanations and gave it a concrete closing sentence instead of a one-line fragment), Section 5
"Verification" (reframed the opening paragraph to state explicitly, before the list, that this
is not six independent proofs), Section 6 "Results" (replaced the bracketed table pointer with
an actual inline table and added a closing sentence contrasting the before/after holdout
result), Section 7 "Discussion" (reviewed, no changes needed), and Section 1 "Introduction"
(tightened Contribution #2's phrasing to match Section 5; added the missing Good (2005)
reference, which was cited in-text but absent from the bibliography).

Still open before submission: exact author/affiliation details and target-venue decision; final
manual confirmation of the decomposition-leakage preprint's author; one last cross-check of
Section 3's numeric tables against README.md immediately before submission, since the README
remains the living source of truth and this file is a periodically-updated snapshot of it.*
