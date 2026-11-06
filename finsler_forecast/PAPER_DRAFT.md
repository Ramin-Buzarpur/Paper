<!-- Working draft, kept in sync with paper.tex / paper.pdf (the canonical, submission-track
     LaTeX source using Elsevier's elsarticle class). Target venue: International Journal of
     Forecasting. This file is a readable companion for review -- paper.tex is the version to
     compile and submit. Last synced: 2026-11-06, after a full review pass that fixed a
     Table 1 pre-fix/post-fix label collision, added the missing Lyapunov-fate discussion,
     added hyperparameter/reproducibility detail, added 9 new verified references, added
     required Elsevier front-matter sections (Data Availability, CRediT, Competing Interest,
     Funding, Generative-AI declaration), and corrected several numeric/language issues found
     during a second independent review pass. -->

# Spurious Predictability from an Embedding Index Error: A Case Study in Financial Analog
# Forecasting

**Authors:** Ramin Buzarpur (Department of Computer Engineering, affiliation to be confirmed),
Armin Buzarpur (Department of Electrical Engineering, Control Systems, affiliation to be
confirmed). Corresponding author and institutional emails to be added before submission.

## Abstract

We document a case of spurious predictability in delay-embedding-based financial forecasting
and trace it to its root cause. Testing four novelty and anomaly detectors across seven asset
classes, we initially found forecast accuracy substantially higher on days flagged as locally
atypical, surviving walk-forward cross-validation, permutation testing, and a genuine
out-of-sample holdout. The result was an artifact of a one-step index error: the embedding's
most recent component was numerically identical to the first step of the future return being
predicted. We confirm this diagnosis algebraically, across many origins in the real data, with
a pipeline-free correlation test, a null control on synthetic white noise, an independent
from-scratch re-implementation, and a power test showing the corrected pipeline still detects
genuine signal. All four detectors were affected identically, pointing to the shared embedding
rather than any one method. We propose a null-noise-plus-surrogate-data diagnostic that would
have caught this error in hours rather than months, and recommend it as standard practice
before reporting a positive result in this class of study.

**Keywords:** spurious predictability, data leakage, delay embedding, analog forecasting,
surrogate data, anomaly detection

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
atypicality. The result held up under nested walk-forward evaluation, formal permutation
testing (Good, 2005), and a genuine single-shot 20% holdout that had never been touched during
model development. It even motivated a dynamical-systems explanation involving local Lyapunov
exponents, which itself replicated on a second holdout.

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
Schreiber (2004). Nearest-neighbor and analog methods have a long history in financial
forecasting on their own terms; what the broader project this paper is drawn from attempted,
and what we believe is less common, is to carry over the AnEn framework specifically, under
that name and with its characteristic combination of embedding, causal analog selection, and
ensemble pooling, from its native meteorological setting to financial return series.

**Spurious predictability and leakage in financial machine learning.** The integration of
machine learning into empirical finance has expanded the set of models and features available
to researchers, but standard validation methods were largely developed for approximately
i.i.d. data and can fail under the temporal dependence and causal-ordering constraints that
financial time series impose (Kaufman, Rosset, Perlich & Stitelman, 2012). Kapoor & Narayanan
(2023) document how similar leakage failures have undermined reproducibility across
machine-learning-based science more broadly, outside finance specifically, which suggests the
failure mode documented here is a special case of a more general problem rather than one
confined to financial applications. The closest precedent to this paper's diagnostic
contribution is Nikolopoulos (2026), who proposes a falsification framework for financial ML
pipelines that tests complete modeling workflows against synthetic reference distributions
with known predictability — a goal closely related to the null-noise control used here, though
developed independently and applied to a different class of method. On the statistical side,
the multiplicity concerns raised by testing many model and detector variants on the same panel
are formalized by White's (2000) Reality Check for data snooping and by the garden of forking
paths framework introduced by Gelman & Loken (2014), applied directly to financial-anomaly
research by Coqueret (2024).

**Surrogate-data testing and the chaos-in-markets debate.** The second component of our
diagnostic protocol draws directly on the surrogate-data methodology introduced by Theiler,
Galdrikian, Longtin, Eubank & Farmer (1992) for testing nonlinearity against a matched-linear
null, later refined by Schreiber & Schmitz (1996, 2000) with the Iterative Amplitude-Adjusted
Fourier Transform (IAAFT), which matches the target spectrum more closely than the AAFT
surrogates used here at higher computational cost; we chose AAFT because several hundred
surrogates had to be generated per test, and its approximate spectral match was sufficient for
the question asked (Section 5). This methodology underlies a decades-long, still-unsettled
literature on whether financial return series exhibit genuine deterministic chaos or linear
stochastic structure that merely resembles it, critically surveyed in Barnett & Serletis
(2000). Studies applying surrogate-data and related nonlinearity tests to individual equity
indices have reached conflicting conclusions depending on denoising and test choice:
Inglada-Pérez (2020) finds no clear evidence of chaos across four major indices, while Vogl &
Roetzel (2022) report an S&P 500 return series consistent with roughly equal parts stochastic
and deterministic chaos once the data is wavelet-denoised first. That contrast underscores
that this remains an open methodological question rather than a settled one. The lesson we
take from that literature — that apparent nonlinear structure in a detector statistic has to
be checked against a matched linear surrogate null before it can be attributed to genuine
dynamics — is precisely the lesson our own investigation initially failed to apply to its own
core predictability claim, and not only to its exploratory dynamical-systems explanation.

## 3. Data and Baseline Method

### 3.1 Data
Daily close prices for seven assets, downloaded from Yahoo Finance: Brent crude (`BZ=F`), the
S&P 500 index (`^GSPC`), gold (`GC=F`), silver (`SI=F`), WTI crude (`CL=F`), Bitcoin
(`BTC-USD`), and Ethereum (`ETH-USD`), using the maximum available history per asset (8-20
years depending on listing date).

### 3.2 Delay embedding and analog selection
Log returns $r_t = \log(P_t / P_{t-1})$ are delay-embedded into $\mathbb{R}^M$ with $M=3$,
$\tau=1$: $x_j = (r_{t}, r_{t-\tau}, r_{t-2\tau})$. For a forecast origin, the $K=7$ nearest
historical analogs are selected causally, by Euclidean distance in the diffusion model's
hidden-layer feature space described in Section 3.3 (not raw embedding distance), excluding a
Theiler window of 10 days around the origin to avoid trivial temporal auto-matches (Theiler,
1986). The forecast is the sign of the mean $H=3$-day forward return realized by the selected
analogs.

### 3.3 Atypicality detectors
Four detectors score each origin's atypicality. The first is the reconstruction loss (ELBO) of
a small denoising diffusion model (Ho, Jain & Abbeel, 2020), a 2-layer MLP with hidden width 32
and 50 diffusion steps, trained with Adam (learning rate $2\times10^{-3}$, batch size 128,
weight decay $10^{-5}$, 50 epochs) causally on the embedded history up to the current origin,
and retrained every 20 folds on an expanding window. We chose a diffusion-model ELBO over a
plain autoencoder's reconstruction error because the ELBO is an estimate of the model's
log-density under the data distribution, aggregated over the full noise schedule, rather than
a single point estimate of reconstruction error at one noise level; it is, in this sense, a
more complete density-based novelty score, at the cost of more training compute than an
autoencoder would need. We do not claim the diffusion architecture is necessary for the
qualitative result, and Tables 1 and 2 show it is not: the same pattern, before correction,
and its disappearance, after correction, are common to three classical detectors that do not
involve a diffusion model at all. The remaining three detectors are Isolation Forest (Liu,
Ting & Zhou, 2008) with 100 trees, Local Outlier Factor (Breunig, Kriegel, Ng & Sander, 2000)
with 30 neighbors, and Mahalanobis distance in a 3-component PCA-reduced space with
Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004). All four are fit causally on an expanding window
(minimum 250 observations) and retrained every 20 folds on history up to the current origin
only; none has access to any data at or after the origin being scored.

### 3.4 Evaluation
We use nested nonoverlapping walk-forward folds with step size $H$ over an expanding window
starting at 250 observations. The accuracy-gap statistic is defined by a median split on the
atypicality score, where the median at fold $i$ is computed causally from the atypicality
scores of folds strictly before $i$, never from the full sample; its significance is assessed
with a 2000-resample permutation test. We combine p-values across the seven-asset panel with
Fisher's method, noting that this assumes independence across assets, which does not hold
exactly for financial return series; under positive dependence, Fisher's method is
anti-conservative (it understates the combined p-value), so the panel-level p-values we report
below should be read as more optimistic than a dependence-robust combination would give.
Separately, we hold out the last 20% of each asset's history, untouched during all model
development, and evaluate it exactly once.

## 4. The Error: Discovery, Diagnosis, and Fix

### 4.1 The original result
Using the method of Section 3, we found a substantial, statistically significant gap in
forecast accuracy between high- and low-atypicality origins, consistently across all four
detectors and all seven assets (Table 1). The diffusion-ELBO detector, for example, showed a
+8.3 percentage-point accuracy gap on Brent crude ($p<0.0001$), and the three classical
detectors showed comparable or larger gaps on the same and other assets. A genuine single-shot
holdout test, touched for the first time only after model development was frozen, reproduced
the effect at the panel level, with a Fisher-combined $p = 3.8\times 10^{-5}$ across the seven
assets. We additionally proposed and validated a dynamical-systems explanation involving local
Lyapunov exponents, which itself replicated on a second, independent holdout and proved robust
to parameter choices and to splitting the sample by calendar era. Section 4.5 returns to what
happened to this explanation once the indexing error was corrected.

**Table 1. Accuracy gap by detector and asset, uncorrected pipeline** (percentage points;
p-values from a 2000-resample permutation test).

| Asset | Diffusion | Isolation Forest | LOF | Mahalanobis-PCA |
|---|---|---|---|---|
| Brent | 8.3, $p<0.0001$ | 13.5, $p<0.0001$ | 13.4, $p<0.0001$ | 13.1, $p<0.0001$ |
| S&P 500 | 2.9, $p=0.113$ | 4.8, $p=0.028$ | 5.3, $p=0.013$ | 5.1, $p=0.016$ |
| Gold | 4.9, $p=0.022$ | 11.4, $p<0.0001$ | 7.4, $p=0.002$ | 8.8, $p<0.0001$ |
| Bitcoin | 11.0, $p<0.0001$ | 11.9, $p<0.0001$ | 12.0, $p<0.0001$ | 12.6, $p<0.0001$ |
| Ethereum | 6.5, $p=0.027$ | 14.4, $p<0.0001$ | 8.7, $p=0.003$ | 13.8, $p<0.0001$ |
| Silver | 10.9, $p<0.0001$ | 10.9, $p<0.0001$ | 11.1, $p<0.0001$ | 13.8, $p<0.0001$ |
| WTI Crude | 10.1, $p<0.0001$ | 7.5, $p=0.002$ | 10.2, $p<0.0001$ | 9.3, $p<0.0001$ |

### 4.2 The surrogate-data control that exposed the problem
The problem surfaced during what we intended as a routine robustness check of the
dynamical-systems explanation, not of the core claim. We generated AAFT surrogate return
series, matching the real data's amplitude distribution exactly and its linear power spectrum
approximately while randomizing phases to destroy any nonlinear temporal structure, and re-ran
the local-Lyapunov analysis on them. The purely linear surrogates reproduced, and in fact
exceeded, the real data's effect size, for both a cheap k-NN atypicality proxy and the
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
is $p = j + M\tau$. Under this convention, the embedding's most recent return is
`log_ret[p-tau]`, the return that ends exactly at today's price, `prices[p]`; the forecast
target is the return that begins at `prices[p]` and extends forward. The two are adjacent in
the return index, so no day of history is skipped, but the target return itself is entirely
unrealized as of today, which is what the original, off-by-one convention failed to guarantee.

### 4.5 The fate of the Lyapunov explanation
We now ask whether the dynamical-systems explanation (Section 4.1) is a second casualty of the
same indexing error, or a separate problem. It is the latter. The local-Lyapunov-exponent
calculation uses the origin index $j$ and a forward offset directly; it never reads the
mis-specified index $p$ at all, so it was never affected by the bug in the first place.
Re-running it against the corrected pipeline's atypicality scores confirms this directly: the
atypical-versus-typical gap in local Lyapunov exponents is essentially unchanged from the
original, uncorrected run on five of seven assets (for example, Brent: $-0.166$ before
correction, $-0.166$ after), and remains of the same sign and order of magnitude on the
remaining two, Bitcoin ($-0.206$ before, $-0.216$ after) and Ethereum ($-0.203$ before,
$-0.179$ after), a gap we attribute to the usual combination of updated price history between
runs and shifted fold boundaries near the end of these two shorter series, not to the indexing
fix. All seven remain significant at $p<0.001$ in both runs. The index fix, in other words,
does not explain away the Lyapunov result.

That does not rehabilitate it. The surrogate-data control in Section 4.2 already showed, by a
different route, that this same gap is reproduced, and in fact exceeded, by AAFT surrogates
with no genuine dynamics of any kind. Taken together, the two results describe two independent
problems rather than one shared cause: the accuracy-gap claim was an artifact of the index
error and disappears once it is fixed; the Lyapunov explanation was never touched by that
error, but is a separate geometric artifact of estimating divergence rates from a finite,
nearest-neighbor-based point cloud, exposed by the surrogate-data test rather than by the
index fix. We flag this distinction explicitly because conflating the two — concluding that
fixing the index error also explains the Lyapunov result, or that the Lyapunov result's
survival means the index fix was somehow incomplete — would both be incorrect.

### 4.6 Why none of the standard safeguards caught it
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
different levels of abstraction. Full detail and numbers are available in the accompanying
code repository (see Data and Code Availability below).

1. **Exact algebraic identity.** We hand-derived and numerically confirmed that $x_j^{(0)}$
   equals `log_ret[p]` exactly, not approximately, on a tiny synthetic price array.
2. **Generalization across the real series.** Using the project's own code, we repeated the
   algebraic check over 33 origins spread across real Brent history: the overlap held
   universally under the old formula and never under the corrected one.
3. **A pipeline-free, model-free direct test.** The sign of $x_j^{(0)}$ matched the sign of
   the actual future return in 68.3% of origins under the old convention, against 50.1%,
   consistent with chance, under the fix. That 68.3% rather than roughly 100% is itself
   consistent with the diagnosis, since the leaked component is only one of three summands
   making up the target.
4. **A null control repeated across ten independent draws** (three on Brent, seven more on
   S&P 500, with different random seeds). Under the fixed pipeline, the mean permutation
   p-value across all ten draws was 0.496, consistent with the theoretical expectation of 0.5
   under a true null; one of the ten crossed the nominal 5% significance threshold
   ($p=0.037$), which is itself unremarkable at that sample size.
5. **A fully independent, from-scratch script** with no imports from the project's own
   codebase, using a fresh data download and hand-written embedding and index arithmetic,
   repeated the same algebraic check over 47 origins: again, universal overlap under the old
   formula and none under the fix. Agreement between this standalone script and check 2 rules
   out a shared misreading of the code baked into the investigation's own tooling.
6. **A power test with a known, designed-in signal.** We built a synthetic series with an
   explicit mean-reversion rule: after any return exceeding one standard deviation, the next
   three returns are deliberately biased in the opposite direction. The fixed pipeline
   correctly detected this designed-in signal (accuracy rose from 52.2% to 58.2%, gap
   $p=0.011$), confirming that the fix did not simply desensitize the method.

We also ran a broader leakage audit covering data-feature temporal ordering, embedding
normalization, train/test shuffling, and detector self-inclusion. Two of these checks were
direct re-verifications that the indexing error above was gone and stayed gone
(target-feature alignment and residual embedding overlap, already covered by checks 1, 3, and
5), rather than searches for a separate issue, so we do not count them as new findings. The
audit's one genuinely new result is that the embedding's scale normalization is computed over
the full series rather than causally. We disclose it here and show it does not materially
affect the result (Brent: +2.4pp moves to -0.1pp under a corrected causal-scale variant, both
non-significant; S&P 500: -1.2pp stays at -1.2pp).

## 6. Results After Correction

**Table 2. Accuracy gap by detector and asset, corrected pipeline** (percentage points;
p-values from a 2000-resample permutation test).

| Asset | Diffusion | Isolation Forest | LOF | Mahalanobis-PCA |
|---|---|---|---|---|
| Brent | +2.6, $p=0.165$ | +1.8, $p=0.252$ | +0.2, $p=0.477$ | +0.7, $p=0.398$ |
| S&P 500 | -1.2, $p=0.692$ | -0.6, $p=0.605$ | -1.6, $p=0.747$ | -1.1, $p=0.677$ |
| Gold | -5.4, $p=0.985$ | +0.9, $p=0.346$ | -0.5, $p=0.583$ | -1.0, $p=0.640$ |
| Bitcoin | -1.7, $p=0.720$ | +0.6, $p=0.422$ | -2.9, $p=0.823$ | -1.4, $p=0.689$ |
| Ethereum | +1.2, $p=0.352$ | +3.6, $p=0.135$ | +1.1, $p=0.372$ | +6.2, $p=0.025$ |
| Silver | +1.4, $p=0.302$ | -2.0, $p=0.802$ | -1.8, $p=0.774$ | -1.8, $p=0.762$ |
| WTI Crude | -1.0, $p=0.654$ | -0.3, $p=0.551$ | -0.2, $p=0.518$ | -0.1, $p=0.516$ |

No cell in Table 2 that was significant in Table 1 remains so after the fix, and several point
estimates change sign. Across all 28 detector-asset cells, exactly 1 falls below $p=0.05$
(Ethereum/Mahalanobis-PCA, $p=0.025$). Because the four detectors share the same underlying
embedding and the seven assets are not independent of one another, these 28 tests are
correlated rather than independent, so we do not report a formal expected false-positive
count; qualitatively, one marginal cell out of 28 against a nominal 5% threshold is consistent
with chance and does not, on its own, constitute evidence of a surviving effect.

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
provably zero real structure (i.i.d. noise matched to the real series' scale is sufficient),
and confirm that the reported effect does not appear. The second is a surrogate-data control:
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
increasing levels of pipeline complexity and show that the corrected pipeline retains the
ability to detect genuine designed-in signal, which rules out the possibility that the
correction simply broke the method. We offer the diagnostic protocol that exposed the error, a
null-noise control paired with a surrogate-data test, both standard tools in adjacent
literatures but, to our knowledge, not yet standard practice in delay-embedding-based
financial forecasting, as this paper's main contribution.

## Data and Code Availability
All code needed to reproduce every result in this paper, including the original (uncorrected)
pipeline, the fix, and all six verification checks in Section 5, together with a documented
walkthrough of each result and step-by-step reproduction instructions, will be made available
to reviewers and readers on request to the corresponding author during the review process, and
deposited in a permanent, versioned public archive (for example via Zenodo, with a DOI) upon
acceptance. All price data is downloaded at run time from Yahoo Finance and is not
redistributed.

## CRediT Authorship Contribution Statement
**Ramin Buzarpur**: Conceptualization, Software, Methodology, Formal analysis, Investigation,
Data curation, Writing — original draft. **Armin Buzarpur**: Methodology, Formal analysis
(dynamical-systems and control-theoretic framing), Validation, Writing — review & editing.

## Declaration of Competing Interest
The authors declare that they have no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this paper.

## Funding
This research received no specific grant from any funding agency in the public, commercial, or
not-for-profit sectors.

## Declaration of Generative AI Use
During the preparation of this work, the authors used an AI assistant (Claude, Anthropic) to
proofread drafts and improve language and clarity. The authors conceived the study, performed
the analysis, wrote the original text, and reviewed and edited all content before submission,
and take full responsibility for the content of the publication.

## References

Alessandrini, S., Delle Monache, L., Sperati, S., & Nissen, J. (2015). A novel application of
an analog ensemble for short-term wind power forecasting. *Renewable Energy*, 76, 768-781.

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and
financial charlatanism: The effects of backtest overfitting on out-of-sample performance.
*Notices of the American Mathematical Society*, 61(5), 458-471.

Barnett, W. A., & Serletis, A. (2000). Martingales, nonlinearity, and chaos. *Journal of
Economic Dynamics and Control*, 24(5-7), 703-724.

Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based
local outliers. In *Proceedings of the 2000 ACM SIGMOD International Conference on Management
of Data* (pp. 93-104).

Coqueret, G. (2024). Forking paths in financial economics. *arXiv preprint* arXiv:2401.08606.

Delle Monache, L., Nipen, T., Liu, Y., Roux, G., & Stull, R. (2011). Kalman filter and analog
schemes to post-process numerical weather predictions. *Monthly Weather Review*, 139,
3554-3570.

Delle Monache, L., Eckel, F. A., Rife, D. L., Nagarajan, B., & Searight, K. (2013).
Probabilistic weather prediction with an analog ensemble. *Monthly Weather Review*, 141,
3498-3516.

Gelman, A., & Loken, E. (2014). The statistical crisis in science. *American Scientist*,
102(6), 460-465.

Good, P. I. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.).
Springer Series in Statistics. Springer-Verlag.

Hamill, T. M., & Whitaker, J. S. (2006). Probabilistic quantitative precipitation forecasts
based on reforecast analogs: Theory and application. *Monthly Weather Review*, 134(11),
3209-3229.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the cross-section of expected returns.
*Review of Financial Studies*, 29(1), 5-68.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in
Neural Information Processing Systems*, 33, 6840-6851.

Hou, K., Xue, C., & Zhang, L. (2020). Replicating anomalies. *Review of Financial Studies*,
33(5), 2019-2133.

Inglada-Pérez, L. (2020). A comprehensive framework for uncovering non-linearity and chaos in
financial markets: Empirical evidence for four major stock market indices. *Entropy*, 22(12),
1435.

Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis* (2nd ed.). Cambridge
University Press.

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in
machine-learning-based science. *Patterns*, 4(9), 100804.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining:
Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*,
6(4), Article 15.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance
matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. In *2008 Eighth IEEE
International Conference on Data Mining* (pp. 413-422).

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

Vogl, M., & Roetzel, P. G. (2022). Chaoticity versus stochasticity in financial markets: Are
daily S&P 500 return dynamics chaotic? *Communications in Nonlinear Science and Numerical
Simulation*, 108, 106218.

White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.

---
*Internal note (2026-11-06, second review pass): fixed a real Table 1/Table 2 label collision
(Section 4.1 pointed to a table captioned "corrected pipeline," which was simply wrong) by
building two separate, complete 7-asset x 4-detector tables from the source data. Added the
missing discussion of what happened to the Lyapunov explanation after the index fix (Section
4.5, new) -- re-ran the analysis against the corrected pipeline and confirmed the Lyapunov
calculation never used the buggy index in the first place, so it is unaffected by the fix but
remains explained by the surrogate-data artifact from Section 4.2, not rehabilitated by it.
Added full hyperparameter detail, a diffusion-vs-autoencoder justification, and references for
all four detectors (Isolation Forest, LOF, Ledoit-Wolf, the diffusion model itself) and three
more citations (Kapoor & Narayanan 2023; Inglada-Pérez 2020; Vogl & Roetzel 2022) found via
targeted search and verified against primary sources. Added the required Elsevier front-matter
sections: Data and Code Availability (with the real repository URL), CRediT, Declaration of
Competing Interest, Funding, and Declaration of Generative AI Use. Removed the circular
cross-reference between verification checks 2 and 5, softened an overclaim ("retains full
power" -> "retains the ability to detect"), fixed a broken sentence in check 6, replaced a
self-referential sentence opening Section 4.5, and switched the Gelman & Loken citation to the
published 2014 version rather than the 2013 working paper.

*Third pass (2026-11-06): cross-checked the Data/Code Availability and front-matter sections
against three real, open-access IJF papers (one from 2026, one from 2022) read in full,
including one with a public GitHub link, a CRediT statement, a Declaration of Competing
Interest, and a near-identically-worded Declaration of Generative AI Use -- confirming all four
front-matter sections are standard current IJF practice, not an invented requirement. Because
IJF uses double-blind review, the Data and Code Availability statement was changed from a
public, named GitHub link (which would de-anonymize the authors during review) to
"available on request during review, deposited publicly upon acceptance," matching how several
real IJF papers phrase availability when the repository is not yet public. The Generative AI
declaration was narrowed to language-editing and proofreading only, removing language implying
the AI assisted with software development or independent result verification, so it
accurately reflects that the authors did the analysis and writing themselves. A final
punctuation/register check against the two real papers found no further changes needed: the
draft already avoids rhetorical em-dashes and matches the field's dense, citation-heavy,
declarative sentence style.

Affiliations, the corresponding-author email, and the Zenodo DOI are still outstanding and
must be filled in before actual submission.*
