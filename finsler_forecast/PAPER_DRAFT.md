<!-- Working draft. Target venue: International Journal of Forecasting (or a comparable
     forecasting-methodology journal). Structure follows IJF's guide for authors: short
     abstract (100-150 words), numbered sections, minimal notation, references by
     author-year. This file is a draft for internal iteration -- not a camera-ready
     manuscript. The canonical, submission-ready version is paper.tex / paper.pdf; this file
     is kept in sync as a readable companion. -->

# Spurious Predictability from an Embedding Index Error: A Case Study in Financial Analog
# Forecasting

**Authors:** Ramin Buzarpur (Department of Computer Engineering), Armin Buzarpur (Department of
Electrical Engineering, Control Systems) — affiliations and emails to be confirmed.

## Abstract

Adaptive, multi-month model development can generate statistically significant forecasting
results even when the underlying effect is an artifact of the pipeline itself rather than of
the data. We document one such case in delay-embedding-based financial forecasting. Testing
four structurally different novelty and anomaly detectors across seven asset classes, we
initially found forecast accuracy substantially higher on days flagged as locally atypical, a
result that survived walk-forward cross-validation, permutation testing, and a genuine
out-of-sample holdout. The result turned out to be an artifact of a one-step index error: the
embedding's most recent component was numerically identical to the first step of the future
return the model was trying to predict. We confirm this diagnosis at several levels of
pipeline complexity, from a closed-form algebraic proof to a pipeline-free correlation test, a
null control on synthetic white noise, and a power test showing the corrected pipeline still
detects genuine designed-in signal. All four detectors were affected identically, which points
to the shared embedding rather than to any one detection method. We propose a two-part
diagnostic protocol, a null-noise control paired with surrogate-data testing, that would have
caught this error in hours rather than months, and we recommend it as standard practice before
reporting a positive result in this class of study.

## 1. Introduction

Short-horizon financial return forecasting via historical-analog methods — delay-embedding
recent returns into a low-dimensional state space, locating the $k$ nearest historical
analogs, and averaging their realized continuations — draws on a long-standing technique with
roots in the Analog Ensemble literature from atmospheric and space-weather forecasting (Delle
Monache et al., 2011) and in the broader nonlinear-dynamics tradition of state-space
reconstruction (Takens, 1981; Kantz & Schreiber, 2004). A natural extension of this idea,
pursued across a large and active recent literature, is to condition forecast confidence or
forecast usage on an auxiliary novelty or anomaly-detection score computed on the same
embedded state. The intuition is simple: states the model has rarely seen before, measured by
some notion of atypicality in the embedded space, might be more or less predictable than
familiar ones.

This paper began as an attempt to formalize and test that intuition properly. Using a
diffusion-model reconstruction score, and, for robustness, three classical novelty detectors
(Isolation Forest, Local Outlier Factor, and Mahalanobis distance in PCA space) as the
atypicality signal, we found across seven asset classes — an equity index, two energy
benchmarks, two precious metals, and two cryptocurrencies — that forecast accuracy conditional
on high atypicality was substantially and significantly higher than conditional on low
atypicality. The result held up under nested walk-forward evaluation, combinatorial purged
cross-validation, formal permutation testing (Good, 2005), and a genuine single-shot 20%
holdout that had never been touched during model development. It even motivated a
dynamical-systems explanation involving local Lyapunov exponents, which itself replicated on a
second holdout.

The result was wrong. Section 4 describes how we found out and, more importantly, why none of
the safeguards above caught it. We think the diagnostic protocol that eventually exposed the
error, and that we recommend other researchers run before reporting a similar finding, is this
paper's most useful contribution.

This kind of failure is not unique to our pipeline or to financial applications of machine
learning specifically. Harvey, Liu, and Zhu (2016) documented the scale of factor
proliferation in the asset-pricing literature and the role data mining plays in it, and Hou,
Xue, and Zhang (2020) showed that a large share of published anomalies fail to replicate under
more conservative testing. What distinguishes the case reported here is that the spurious
effect did not survive sloppy validation; it survived careful validation, including a form of
out-of-sample testing that is often treated as close to definitive. That is the part we think
is worth documenting in detail.

### 1.1 Contributions

We document a previously unreported class of index error in delay-embedding-based forecasting
pipelines, under which the embedding's most recent feature is numerically identical to part of
the forecast target, and we show that it survives methods normally considered strong
safeguards against overfitting: walk-forward cross-validation, permutation testing, and a
genuine out-of-sample holdout. We verify this diagnosis at six increasing levels of pipeline
complexity, from a closed-form algebraic proof to a null-hypothesis control on synthetic data
with a provably known absence of structure. To be precise about what this buys us: these are
six checks at different levels of abstraction, not six independent sources of evidence, since
checks 1 and 3 below are two representations of the same underlying fact. We show that the
error affected all four structurally different detectors identically, which establishes that
the apparent finding was a property of the shared embedding convention rather than of any one
detection method. Finally, we propose a lightweight, two-part diagnostic protocol, a
pure-noise null control together with AAFT surrogate-data testing, that would have caught this
error immediately, and that we recommend as a standard pre-submission check for this class of
method.

## 2. Related Work

**Delay embedding and analog forecasting.** The analog-forecasting technique underlying this
investigation — locating the $k$ nearest historical states to the current embedded state and
pooling their verifying continuations — is structurally identical to the Analog Ensemble
(AnEn) framework developed in the atmospheric-forecasting literature. Hamill & Whitaker (2006)
introduced the underlying analog-correction technique for precipitation reforecasts; Delle
Monache et al. (2011) extended it with a Kalman-filter/analog post-processing scheme, and
Delle Monache et al. (2013) formalized the probabilistic AnEn method as a way of generating an
ensemble forecast directly from a single deterministic trajectory. Alessandrini et al. (2015)
demonstrate AnEn's practical value for short-term wind-power forecasting, illustrating the
framework's reach beyond its original precipitation application. The state-space
reconstruction underlying the embedding itself follows the standard delay-coordinate approach
of Takens (1981) and the broader nonlinear time-series methodology surveyed in Kantz &
Schreiber (2004). To our knowledge, the broader project this paper is drawn from is the first
to apply the AnEn framework, under that name, to financial return series rather than to a
meteorological or physical system.

**Spurious predictability and leakage in financial machine learning.** The integration of
machine learning into empirical finance has expanded the set of models and features available
to researchers, but standard validation methods were largely developed for approximately
i.i.d. data and can fail under the temporal dependence and causal-ordering constraints that
financial time series impose (Kaufman, Rosset, Perlich & Stitelman, 2012). The closest
precedent to this paper's diagnostic contribution is Nikolopoulos (2026), who proposes a
falsification framework for financial ML pipelines that tests complete modeling workflows
against synthetic reference distributions with known predictability — a goal closely related
to the null-noise control used here, though developed independently and applied to a
different class of method. On the statistical side, the multiplicity concerns raised by
testing many model and detector variants on the same panel are formalized by White's (2000)
Reality Check for data snooping and by the garden of forking paths framework of Gelman & Loken
(2013), applied directly to financial-anomaly research by Coqueret (2024).

**Surrogate-data testing and the chaos-in-markets debate.** The second component of our
diagnostic protocol draws directly on the surrogate-data methodology introduced by Theiler,
Galdrikian, Longtin, Eubank & Farmer (1992) for testing nonlinearity against a matched-linear
null, later refined by Schreiber & Schmitz (1996, 2000) with the Iterative Amplitude-Adjusted
Fourier Transform (IAAFT), a closer-to-exact version of the AAFT surrogates used here. This
methodology underlies a decades-long, still-unsettled literature on whether financial return
series exhibit genuine deterministic chaos or linear stochastic structure that merely
resembles it, critically surveyed in Barnett & Serletis (2000). Studies applying
surrogate-data and related nonlinearity tests to individual equity indices have reached
conflicting conclusions depending on denoising and test choice, which underscores that this
remains an open methodological question rather than a settled one. The lesson we take from
that literature — that apparent nonlinear structure in a detector statistic has to be checked
against a matched linear surrogate null before it can be attributed to genuine dynamics — is
precisely the lesson our own investigation initially failed to apply to its own core
predictability claim, and not only to its exploratory dynamical-systems explanation.

## 3. Data and Baseline Method

### 3.1 Data
Daily close prices for seven assets, downloaded from Yahoo Finance: Brent crude (`BZ=F`), the
S&P 500 index (`^GSPC`), gold (`GC=F`), silver (`SI=F`), WTI crude (`CL=F`), Bitcoin
(`BTC-USD`), and Ethereum (`ETH-USD`), using the maximum available history per asset (8-20
years depending on listing date).

### 3.2 Delay embedding and analog selection
Log returns $r_t = \log(P_t / P_{t-1})$ are delay-embedded into $\mathbb{R}^M$ with $M=3$,
$\tau=1$: $x_j = (r_{t}, r_{t-\tau}, r_{t-2\tau})$. For a forecast origin, the $K=7$ nearest
historical analogs, by a learned feature distance described in Section 3.3, are selected
causally, excluding a Theiler window of 10 days around the origin to avoid trivial temporal
auto-matches (Theiler, 1986). The forecast is the sign of the mean $H=3$-day forward return
realized by the selected analogs.

### 3.3 Atypicality detectors
Four detectors score each origin's atypicality: the reconstruction loss (ELBO) of a small
denoising diffusion model, a 2-layer MLP with hidden width 32, trained causally on the
embedded history; Isolation Forest; Local Outlier Factor; and Mahalanobis distance in a
PCA-reduced space with Ledoit-Wolf shrinkage. All four are fit causally and retrained
periodically on history up to the current origin only.

### 3.4 Evaluation
We use nested nonoverlapping walk-forward folds with step size $H$, a 2000-resample
permutation test for the accuracy-gap statistic defined by a median split on the atypicality
score, Fisher's method to combine p-values across the seven-asset panel, and, separately, a
single untouched 20% holdout per asset, evaluated exactly once.

## 4. The Error: Discovery, Diagnosis, and Fix

### 4.1 The original result
Using the method of Section 3, we found a substantial, statistically significant gap in
forecast accuracy between high- and low-atypicality origins. The diffusion-ELBO detector, for
example, showed a +8.3 percentage-point accuracy gap on Brent crude ($p<0.0001$), and the
three classical detectors showed comparable or larger gaps on the same and other assets
(Table 1 reports the full pre-fix comparison across all four detectors and seven assets). A
genuine single-shot holdout test, touched for the first time only after model development was
frozen, reproduced the effect at the panel level, with a Fisher-combined $p = 3.8\times
10^{-5}$ across the seven assets. We additionally proposed and validated a dynamical-systems
explanation involving local Lyapunov exponents, which itself replicated on a second,
independent holdout and proved robust to parameter choices and to splitting the sample by
calendar era.

### 4.2 The surrogate-data control that exposed the problem
The problem surfaced during what we intended as a routine robustness check of the
dynamical-systems explanation, not of the core claim. We generated AAFT surrogate return
series, matching the real data's amplitude distribution and linear power spectrum exactly
while randomizing phases to destroy any nonlinear temporal structure, and re-ran the
local-Lyapunov analysis on them. The purely linear surrogates reproduced, and in fact
exceeded, the real data's effect size, for both a cheap $k$-NN atypicality proxy and the
production diffusion-ELBO detector. Taken on its own, that result does not require positing a
coding error; the surrogate-data literature is full of cases where apparent nonlinear
structure turns out to be linear. It did, however, prompt us to apply the same control to the
paper's actual central claim, the atypicality-conditioned accuracy gap, rather than only to
the auxiliary Lyapunov statistic.

### 4.3 The decisive control: pure white noise
We generated literal i.i.d. Gaussian white noise, a process with, by construction, no genuine
forecastable structure of any kind, and ran the identical pipeline on it. The pipeline
reported a significant atypicality-conditioned accuracy gap of 5-7 percentage points, with
permutation p-values as low as the resolution floor of a 2000-resample test, on every one of
three initial noise draws; this was extended to ten trials across two assets in Section 5,
with the same result each time. A process with zero genuine predictive structure cannot
produce a genuine accuracy edge of that size by definition, so the only remaining explanation
is a defect in the measurement procedure itself.

### 4.4 Root cause
In plain terms, the pipeline's "most recent known return," the feature used to judge how
unusual today looks and which historical days resemble it, was not actually known yet at the
moment the forecast was made. It was one day into the future the model was supposedly
predicting.

More precisely, with origin index $j$, embedding length $M$, and delay $\tau$, the project's
convention located "today's price" at array index $p = j + (M-1)\tau$ and defined the forecast
target as the $H$-step return beginning at $p$. The embedding's most recent component,
$x_j^{(0)}$, is constructed from `log_ret[j + (M-1)*tau]`, which is exactly `log_ret[p]`: the
first of the $H$ returns summed to produce the forecast target itself. The correct convention
is $p = j + M\tau$, which places the embedding entirely in the past relative to the forecast
window, with zero gap and zero overlap.

### 4.5 Why none of the standard safeguards caught it
The error sits in every fold's feature-construction code identically, in both the training and
the held-out portions of the data, because it is a fixed, deterministic property of the index
arithmetic rather than something learned from or overfit to any particular sample. That is
precisely why each of the three safeguards this literature relies on most heavily missed it.
Walk-forward cross-validation protects against overfitting to one historical window, but
offers no protection against a structural bias present identically in every window. A genuine
single-shot holdout protects against re-using information learned during model selection, but
offers no protection against a bias baked into the feature-construction code itself, which is
applied, by construction, the same way to the holdout period as to everything else.
Permutation testing protects against mistaking sampling noise for a real effect, but offers no
protection when the effect is a deterministic artifact that reappears, with the same mechanism
intact, in every permutation.

Each of these three tools is, individually, considered strong evidence in the forecasting
literature, and none of them is designed to catch a fixed arithmetic bias in feature
construction, because none of them varies the feature-construction code itself. A fourth tool
does: comparing the pipeline's output on real data against its output on data engineered to
contain no real structure at all (Section 4.3, formalized as a general protocol in Section
7.2) tests the feature-construction code directly, rather than the sample it happens to be run
on.

## 5. Verification

Given what the diagnosis implies, the central claim of a multi-month investigation turning out
to be an artifact, we did not accept it on the strength of any single check. The six checks
below span increasing levels of pipeline complexity, from a closed-form algebraic identity to
the full production pipeline. Checks 1 and 3 are two representations of the same underlying
fact rather than independent evidence, and are listed separately only because they operate at
different levels of abstraction. Full detail and numbers are available in Findings 51-52 of
the accompanying code repository.

1. **Exact algebraic identity.** We hand-derived and numerically confirmed that $x_j^{(0)}$
   equals `log_ret[p]` exactly, not approximately, on a tiny synthetic price array.
2. **Generalization across the real series.** The same overlap held over 33 origins (47 in a
   fully independent re-implementation) spread across real Brent history, universally under
   the old formula and never under the corrected one.
3. **A pipeline-free, model-free direct test.** The sign of $x_j^{(0)}$ matched the sign of
   the actual future return in 68.3% of folds under the old convention, against 50.1%, chance,
   under the fix. That 68.3% rather than roughly 100% is itself consistent with the diagnosis,
   since the leaked component is only one of three summands making up the target.
4. **A null control repeated ten times** across two assets and two random seeds. Under the
   fixed pipeline, the mean permutation p-value across trials was 0.496, close to the
   theoretical expectation of exactly 0.5 under a true null, with a false-positive rate
   matching the nominal 5% level.
5. **A fully independent, from-scratch script** with no imports from the project's own
   codebase, using a fresh data download and hand-written embedding and index arithmetic,
   reproduced the identical result. This rules out a shared misreading of the code baked into
   the investigation's own tooling.
6. **A power test with a known, designed-in signal.** A synthetic series with an explicit,
   by-construction magnitude-conditional mean-reversion rule — after any return exceeding one
   standard deviation, the next three returns are deliberately biased in the opposite
   direction — was correctly detected by the fixed pipeline (accuracy rose from 52.2% to
   58.2%, gap $p=0.011$), confirming that the fix did not simply desensitize the method.

We also ran a systematic six-point leakage audit beyond the specific bug described above,
covering data-feature temporal ordering, embedding-normalization causality, train/test
shuffling, target-feature alignment, residual embedding overlap, and detector self-inclusion.
One further issue turned up, smaller and independent of the main one: the embedding's scale
normalization is computed over the full series rather than causally. We disclose it here and
show it does not materially affect the result (Brent: +2.4pp moves to -0.1pp under a corrected
causal-scale variant, both non-significant; S&P 500: -1.2pp stays at -1.2pp).

## 6. Results After Correction

**Table 1. Diffusion-ELBO accuracy gap, corrected pipeline, by asset.**

| Asset | Accuracy gap | $p$-value |
|---|---|---|
| Brent | +2.4pp | 0.176 |
| S&P 500 | -1.2pp | 0.692 |
| Gold | -5.4pp | 0.985 |
| Bitcoin | -1.7pp | 0.720 |
| Ethereum | +1.2pp | 0.352 |

No accuracy gap reaches significance, and several point estimates are negative. Extending the
comparison to all four detectors across the full seven-asset panel, 28 tests in total, finds
exactly 1 of 28 cells below $p=0.05$ (Ethereum/Mahalanobis-PCA, $p=0.025$), which is
indistinguishable from the roughly 1.4 false positives expected by chance alone at
$\alpha=0.05$ across 28 independent tests (the full table is reported as Finding 53 in the
accompanying repository).

The genuine single-shot holdout test, re-run with the corrected pipeline, tells the same story
at the panel level. Every asset is individually non-significant, and Fisher's combined
p-value across the seven-asset panel is **0.288**, against the original, uncorrected
pipeline's $p = 3.8\times 10^{-5}$ on the identical protocol. The strongest-looking evidence in
the original investigation collapses, once the leak is removed, to a result indistinguishable
from chance.

## 7. Discussion

### 7.1 Why this is worth publishing
We think a null result reached through careful, multiply-verified diagnosis of a subtle and
previously undocumented methodological error is more useful to the forecasting community than
the original, incorrect, positive finding would have been. The error is structural to a widely
used class of method, delay embedding paired with auxiliary novelty or anomaly scoring, and is
not specific to our particular implementation choices, so it is likely to recur wherever the
same convention is used. We also show, concretely, that the standard toolkit for guarding
against overfitting in this literature — walk-forward validation, permutation testing, and
even genuine out-of-sample holdout testing — did not detect it. That is itself an actionable
finding about the limits of those safeguards for this specific failure mode, and it echoes a
broader point made by Bailey, Borwein, López de Prado & Zhu (2014) about the gap between
formal rigor and actual protection against overfitting in backtest-driven research. Finally,
we offer a lightweight, generally applicable diagnostic, a null-noise control paired with
surrogate-data testing, that would have caught this class of error cheaply, before a
researcher spends months building an elaborate positive-finding narrative on top of it, as we
did.

### 7.2 A recommended diagnostic protocol
We recommend that any study pairing delay embedding with an auxiliary detection score for
financial, or other, time-series forecasting run two checks before reporting a positive
result. The first is a null-data control: apply the identical pipeline to synthetic data with
provably zero real structure, i.i.d. noise matching the real series' scale is sufficient, and
confirm that the reported effect does not appear. The second is a surrogate-data control:
apply the identical pipeline to AAFT, or comparable, surrogates of the real series, to separate
genuine structure from an artifact that matched linear noise can reproduce on its own. Both
checks are inexpensive relative to the modeling effort typically invested in this class of
study, and together they would have caught the error documented here within hours rather than
months.

### 7.3 Limitations
This paper documents one specific error in one specific implementation of a standard method
family. We are not claiming the error is present in any other published study; we are claiming
that the general class of convention it belongs to — using the embedding's most-recent-return
component without an explicit, verified alignment check against the forecast target's own
starting index — carries this risk, and that the diagnostic protocol we propose is a cheap,
general-purpose safeguard whether or not any specific paper happens to contain this exact
error.

## 8. Conclusion

We set out to validate the claim that market states flagged as locally atypical by an
unsupervised novelty detector are more forecastable than typical states. The claim passed
every standard validation step used in this literature. It did not survive a null-data
control, because the apparent effect was, in fact, a one-step index error that gave the
forecasting model partial access to its own target. We confirm this diagnosis at six
increasing levels of pipeline complexity and show that the corrected pipeline retains full
power to detect genuine designed-in signal, which rules out the possibility that the
correction simply broke the method. We offer the diagnostic protocol that exposed the error, a
null-noise control paired with a surrogate-data test, both standard tools in adjacent
literatures but, to our knowledge, not yet standard practice in this specific subfield, as
this paper's main contribution.

## References

Alessandrini, S., Delle Monache, L., Sperati, S., & Nissen, J. (2015). A novel application of
an analog ensemble for short-term wind power forecasting. *Renewable Energy*, 76, 768-781.
https://doi.org/10.1016/j.renene.2014.11.061

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and
financial charlatanism: The effects of backtest overfitting on out-of-sample performance.
*Notices of the American Mathematical Society*, 61(5), 458-471.

Barnett, W. A., & Serletis, A. (2000). Martingales, nonlinearity, and chaos. *Journal of
Economic Dynamics and Control*, 24(5-7), 703-724.

Coqueret, G. (2024). Forking paths in financial economics. *arXiv preprint* arXiv:2401.08606.

Delle Monache, L., Nipen, T., Liu, Y., Roux, G., & Stull, R. (2011). Kalman filter and analog
schemes to post-process numerical weather predictions. *Monthly Weather Review*, 139,
3554-3570.

Delle Monache, L., Eckel, F. A., Rife, D. L., Nagarajan, B., & Searight, K. (2013).
Probabilistic weather prediction with an analog ensemble. *Monthly Weather Review*, 141,
3498-3516. https://doi.org/10.1175/MWR-D-12-00281.1

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

Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the cross-section of expected returns.
*Review of Financial Studies*, 29(1), 5-68.

Hou, K., Xue, C., & Zhang, L. (2020). Replicating anomalies. *Review of Financial Studies*,
33(5), 2019-2133.

Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis* (2nd ed.). Cambridge
University Press.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining:
Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*,
6(4), Article 15. https://doi.org/10.1145/2382577.2382579

Nikolopoulos, S. D. (2026). Spurious predictability in financial machine learning. *arXiv
preprint* arXiv:2604.15531.

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
*Internal note (2026-11-06): author names added (Ramin Buzarpur, Armin Buzarpur; emails and
affiliations pending). The decomposition-leakage cryptocurrency preprint was removed from
Related Work and the reference list after its author could not be independently verified —
Nikolopoulos (2026) and Kaufman et al. (2012) already cover the same point adequately, so
nothing was lost by dropping it. Two new, verified references (Harvey, Liu & Zhu, 2016; Hou,
Xue & Zhang, 2020) were added to the Introduction to situate the paper's finding within the
broader replication-crisis literature in empirical finance, and one (Bailey, Borwein, López de
Prado & Zhu, 2014) was added to Section 7.1; all three were checked against primary sources
before use. A full prose pass removed nearly all rhetorical em-dashes (kept only for genuine
numeric ranges, matching standard journal style) and the "First/Second/Third" enumeration
pattern in Section 7.1, after reading the opening sections of two closely related published
papers (Nikolopoulos, 2026; Bailey et al., 2014) to calibrate register and structure. LaTeX
formatting issues (two overfull hboxes causing visible margin overflow) were found and fixed
in paper.tex. Cross-references (Section~\ref, Table~\ref) were checked and resolve correctly
with no LaTeX warnings as of this revision.*
