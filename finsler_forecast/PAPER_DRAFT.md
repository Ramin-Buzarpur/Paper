<!-- Working draft. Target venue: International Journal of Forecasting (or a comparable
     forecasting-methodology journal). Structure follows IJF's guide for authors: short
     abstract (100-150 words), numbered sections, minimal notation, references by
     author-year. This file is a draft for internal iteration -- not a camera-ready
     manuscript. -->

# Spurious Predictability from an Embedding Index Error: A Case Study in Financial Analog
# Forecasting

**Authors:** [Author 1, Computer Engineering], [Author 2, Electrical/Control Engineering]

## Abstract (draft, ~150 words)

We document, diagnose, and trace to its root cause a case of spurious predictability in
delay-embedding-based financial forecasting. Across a multi-month investigation spanning four
structurally different novelty/anomaly detectors (a diffusion-model reconstruction score,
Isolation Forest, Local Outlier Factor, and Mahalanobis distance) and seven asset classes, we
initially found that forecast accuracy was substantially higher on days flagged as locally
atypical -- a result that survived nested walk-forward cross-validation, formal permutation
testing, and a genuine single-shot out-of-sample holdout. We show this entire result was an
artifact of a one-step index error in the standard delay-embedding convention, under which the
embedding's most recent component is numerically identical to the first step of the very
future return being predicted. We verify this diagnosis six independent ways, including an
algebraic proof, a pipeline-free correlation test, a null control on synthetic white noise,
and a power test confirming the corrected pipeline retains the ability to detect genuine
designed-in signal. We propose a simple diagnostic protocol -- a null-noise control plus
surrogate-data testing -- that any future embedding-based forecasting study should run before
reporting a positive result.

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
2. We verify the diagnosis through six independent, increasingly rigorous checks, including a
   closed-form algebraic proof and a null-hypothesis control on synthetic data with a provably
   known absence of structure.
3. We show the error affected all four structurally different detectors tested identically,
   establishing that the apparent "finding" was a property of the shared embedding
   convention, not of any one detection method.
4. We propose a lightweight, two-part diagnostic protocol (a pure-noise null control, and
   AAFT surrogate-data testing) that would have caught this error immediately, and that we
   recommend as a standard pre-submission check for this entire class of method.

## 2. Related Work

**Delay embedding and analog forecasting.** [Takens 1981; Kantz & Schreiber 2004; Delle
Monache et al. 2011 -- AnEn]. [cite colleague's engine / prior analog-forecasting literature].

**Spurious predictability and leakage in financial machine learning.** The closest precedents
to this paper's diagnostic contribution are Nikolopoulos (2024/2025), *"Spurious
Predictability in Financial Machine Learning,"* which formalizes a falsification framework for
diagnosing exactly this class of problem; the foundational survey on leakage detection and
avoidance in data mining (Kaufman et al., 2012, ACM TKDD); and a recent demonstration that
decomposition-ensemble cryptocurrency forecasting models' reported accuracy is substantially
attributable to look-ahead bias inherent in two-sided decomposition methods (anonymous
preprint, 2024/2025), which proposes a falsification test structurally similar to the one used
here.

**Surrogate-data testing and the chaos-in-markets debate.** Our diagnostic protocol's second
component draws directly on the decades-long literature testing whether financial return
series exhibit genuine deterministic chaos (Theiler et al., 1992, on surrogate data methods;
and the extensive, still-unsettled empirical literature applying surrogate-data and related
tests to equity indices, e.g. studies on the S&P 500, FTSE 100, and various national indices,
with conflicting conclusions depending on denoising and test choice). This literature's
central methodological lesson -- that apparent nonlinear structure in a detector statistic
must be checked against a matched linear surrogate null before being attributed to genuine
dynamics -- is the lesson this paper's own investigation initially failed to apply to its
*own* core predictability claim, not only to its exploratory dynamical-systems explanation.

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
forecast accuracy between high- and low-atypicality origins. [See Table 1 -- original,
pre-fix numbers, e.g. the original Finding 31/33 table.] A genuine single-shot holdout test,
touched for the first time after all model development was frozen, reproduced the effect at
the panel level (Fisher-combined $p = 3.8\times 10^{-5}$ across seven assets). We additionally
proposed and validated a dynamical-systems explanation (local Lyapunov exponents), which
itself replicated on a second, independent holdout and was robust to parameter choices and to
splitting the sample by calendar era.

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
(0.0000), on every one of three independent noise draws. This result cannot be explained by
anything except a defect in the measurement procedure itself.

### 4.4 Root cause
Tracing the index arithmetic: with origin index $j$, embedding length $M$, and delay $\tau$,
the project's convention located "today's price" at array index $p = j + (M-1)\tau$ and
defined the forecast target as the $H$-step return beginning at $p$. But the embedding's most
recent component, $x_j^{(0)}$, is constructed from $\texttt{log\_ret}[j + (M-1)\tau]$ --
exactly $\texttt{log\_ret}[p]$: the **first of the $H$ returns summed to produce the forecast
target itself**. The feature used to assess how "atypical" today's state is, and to select
which historical analogs look similar, was in fact one step of tomorrow, not a step of
history. The correct convention is $p = j + M\tau$, which places the embedding entirely in the
past relative to the forecast window with zero gap and zero overlap.

### 4.5 Why none of the standard safeguards caught it
The error is baked into every fold's feature construction identically in both the "training"
and the "held-out" portions of the data, because it is a fixed, deterministic property of the
index arithmetic, not a property learned from or overfit to any particular sample. Walk-forward
cross-validation protects against overfitting to a specific historical window; it provides no
protection against a structural bias present uniformly across all windows. A genuine
single-shot holdout protects against re-using information learned during model selection; it
provides no protection against a bias baked into the feature-construction code that is applied
identically, by construction, to the holdout period. Permutation testing protects against
mistaking sampling noise for a real effect; it provides no protection when the "effect" is a
deterministic artifact that appears in every resample with the same mechanism intact. None of
these three standard safeguards -- individually considered strong evidence in the forecasting
literature -- are designed to catch a fixed arithmetic bias in feature construction. A fourth
tool, surrogate/null-data testing, is.

## 5. Verification

Because of the scale of what the diagnosis implies (the central claim of a multi-month
investigation is an artifact), we verified it six independent ways before accepting it (full
detail and numbers: README Findings 51-52 of the accompanying code repository).

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

Table 2 reports the corrected pipeline's results on real market data. [See README Finding 51
for the single-detector table and Finding 53 for the full 4-detector $\times$ 7-asset
comparison and the corrected holdout test.]

- Across the original single-detector test (diffusion ELBO) on 5 assets: no accuracy gap
  reaches significance ($p$ ranges 0.18 to 0.99; several point estimates are negative).
- Across the full 4-detector $\times$ 7-asset comparison (28 tests): 1 of 28 reaches $p<0.05$
  (Ethereum/Mahalanobis, $p=0.025$) -- indistinguishable from the $\approx$1.4 false positives
  expected by chance at $\alpha=0.05$ across 28 independent tests.
- The genuine single-shot holdout test, re-run with the corrected pipeline: every asset
  individually non-significant; Fisher's combined $p$-value across the seven-asset panel is
  **0.288**, versus the original (uncorrected) pipeline's $p=3.8\times 10^{-5}$ on the
  identical protocol.

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
forecasting model partial access to its own target. We verify this diagnosis six independent
ways and show the corrected pipeline retains full power to detect genuine designed-in signal,
ruling out that the correction merely broke the method. We offer the diagnostic protocol that
exposed the error -- a null-noise control and a surrogate-data test, both standard tools in
adjacent literatures but, to our knowledge, not yet standard practice in this specific
subfield -- as this paper's primary, generalizable contribution.

## References

[To be completed with full citations: Takens 1981; Kantz & Schreiber 2004; Delle Monache et
al. 2011; Theiler 1986; Theiler et al. 1992; Good 2005 (permutation tests); Gelman & Loken
2013 (garden of forking paths); Coqueret 2024 (forking paths in financial economics);
Nikolopoulos 2024/2025 (spurious predictability in financial ML); Kaufman et al. 2012 (leakage
in data mining, ACM TKDD); Hyndman & Koehler 2006 (MASE); Diebold & Mariano 1995; the
decomposition-ensemble cryptocurrency leakage preprint; the surrogate-data-vs-chaos-in-markets
literature cited in Section 2.]

---
*Internal note: this draft intentionally omits a literature-complete reference list and exact
author/affiliation details pending a decision on target venue and submission timeline. Section
3 (data/method) and the numeric tables in Sections 4/5/6 should be cross-checked one more time
against README.md immediately before any submission, since the README is the living source of
truth and this draft is a snapshot.*
