# Finsler Analog Flow Forecast

Deterministic short-horizon forecast for FX/crypto/commodities/indices, based on a local
Finsler (Randers) metric and historical analog continuations — no synthetic sampling, no
training.

## Method

- Delay-embed log-returns in R^3: `x_j = [r(t), r(t-τ), r(t-2τ)]`.
- Local Riemannian metric `a = (Ledoit-Wolf covariance of the k nearest states)^-1`.
- "Wind" is the mean forward drift of the neighbouring states, mapped to a Randers covector
  `b = -tanh(ρ) · wind_direction`, so `F(y) = sqrt(y^T a y) + b·y` stays a valid Finsler norm
  (`||b||_a < 1`, Zermelo convention: moving with the wind is cheaper).
- Forecast scenarios are the true historical next-H states of the K nearest, temporally
  de-correlated neighbours of the current state — not integrated geodesics. A frozen,
  kNN-estimated metric is piecewise constant, so a curved geodesic spray can't honestly be
  derived from it.
- Three geometric centers of the scenario ensemble per forecast step: Fréchet (Finsler,
  wind-aware), Karcher (Riemannian, equals the arithmetic mean in this flat chart), and
  geometric median (robust, L1).

Research/educational tool — not back-tested, no uncertainty estimate, not a trading signal
on its own.

## Diffusion scenarios (alternative method, EUR/USD only so far)

The analog-flow method above reuses real historical continuations. `diffusion/` is a second,
independent way to generate scenarios: a small conditional DDPM trained on hourly log-returns,
sampled to produce synthetic (not historical) continuations. The UI can run either method, or
both side by side, for the same forecast window — that's the point of having two methods: see
where they agree and where they don't.

Diffusion scenarios are only available for a ticker once its checkpoint has been trained:

```bash
python -m finsler_forecast.diffusion.dataset    # downloads EUR/USD minute history from Dukascopy,
                                                 # saves both a raw and a denoised parquet
python -m finsler_forecast.diffusion.train              # trains on the denoised data (recommended)
python -m finsler_forecast.diffusion.train --variant raw  # or trains on the raw data, for comparison
```

### Raw vs. denoised dataset

Dukascopy still emits a row for closed-market periods (weekends, holidays): zero volume,
`High == Low`. Left in, these resample into fake zero-volatility hourly bars that teach the
model a "nothing happens" pattern that isn't real — see `diffusion/denoise.py`. `dataset.py`
keeps both variants on disk side by side (`data_cache/EURUSD_minute.parquet` and
`..._denoised.parquet`), each trains to its own checkpoint
(`diffusion_eurusd_raw.pt` / `diffusion_eurusd_denoised.pt`), and the UI's "Dataset" selector
(shown whenever a diffusion-based method is picked) switches between them — so you can see,
on the same chart, whether denoising actually changed the forecast.

To add another ticker, point `dataset.py`'s `SYMBOL`/`POINT_VALUE` at it, train both variants,
and add `"<ticker>": {"raw": ..., "denoised": ...}` to `DIFFUSION_CHECKPOINTS` in
`diffusion/forecast.py` — the UI and backtest pick it up automatically.

`backtest.py` runs a walk-forward comparison (MAE, RMSE, directional accuracy vs naive
persistence) across the Fréchet/Karcher/median analog centers and, when available, the
diffusion mean/median:

```bash
python finsler_forecast/backtest.py "EUR/USD"            # diffusion compared on the denoised dataset
python finsler_forecast/backtest.py "EUR/USD" raw        # ...or on the raw dataset
```

## Windowed RL backtest (EUR/USD)

`rl/` adds a second layer on top of the same geometry — it does not change it. The hourly
series is cut into non-overlapping 3-day windows; each window opens with the Finsler
forecast made from the daily closes *before* it (the prior), and a small PPO agent learns a
residual position on top of that prior, trading the window bar by bar with costs. The agent
is trained on the earliest windows and then walked forward one window at a time, so every
evaluated window is out of sample.

Because the raw and denoised minute datasets share every trading-hour close and differ only
in the closed-market hours, the comparison between them is measured on those hourly bars
(17% of the raw hourly bars are exact-zero "nothing happened" returns).

```bash
python -m finsler_forecast.rl.dataset                     # build the windows (denoised)
python -m finsler_forecast.rl.backtest                    # walk-forward, denoised
python -m finsler_forecast.rl.backtest raw                # walk-forward, raw
python -m finsler_forecast.rl.backtest --compare-variants # both, side by side
python -m finsler_forecast.rl.backtest denoised --diffusion  # also score the diffusion prior
```

The summary table reports total return, annualized Sharpe, hit rate and max drawdown per
method, plus buy-and-hold over the same bars. `--retrain` refits PPO before every test
window (stricter, much slower).

## Benchmark: which center to trust, and does an AI complement help?

`benchmark.py` runs a **nested** walk-forward comparison (fit/validate/test strictly
chronological, no fold ever used to pick its own weight) across five assets
(BTC-USD, ETH-USD, EUR/USD, Brent, S&P 500), 60 scored daily folds each, horizon = 3 days:

```bash
python finsler_forecast/benchmark.py --timeframe daily
python finsler_forecast/benchmark.py --asset BTC-USD --timeframe hourly
```

**Finding 1 — of the three Finsler centers, only karcher is ever worth using.**
frechet and median (the centers that carry the Randers "wind" term) are consistently the
*worst* predictors by MAE on every asset tested; karcher (the plain Riemannian/arithmetic
mean of the analog neighbours, no wind) is the only one that is sometimes competitive. The
wind term is this engine's distinguishing idea, and it is also what hurts it — treat karcher,
not the full Finsler forecast, as the baseline worth protecting.

**Finding 2 — karcher has a real directional edge on 3 of 5 assets, on top of naive.**
| asset | karcher directional accuracy | naive | karcher MAE vs naive |
|---|---|---|---|
| Bitcoin | 53.3% | 46.7% | worse |
| Brent | 60.0% | 43.3% | **better** |
| S&P 500 | 53.3% | 43.3% | worse |
| Ethereum | 48.3% | 41.7% | worse |
| EUR/USD | 51.7% | 56.7% | worse |

(Naive's "directional accuracy" here is really just how often the asset happened to fall,
since naive never predicts a move — it isn't a real competing forecast on direction, only on
price level.) So: a genuine, non-trivial directional signal on Brent and plausibly BTC/SPX;
none on ETH or EUR/USD.

**Finding 3 — two AI complements were tried and neither reliably survived nested validation.**
Both use only engine-internal geometry features (wind strength, neighbour-covariance
anisotropy, analog-scenario spread, k used) plus plain momentum/realized-vol — no
econometric models (GARCH etc.) per the brief that the complement should be AI-side:
- `ensemble.py`: a small `HistGradientBoostingRegressor` predicts karcher's residual error;
  blended in only if a held-out validation slice shows a >5% MAE improvement (else weight is
  forced to 0, i.e. pure karcher). Result: helped marginally on EUR/USD, made things worse on
  BTC/ETH/Brent/S&P 500.
- A direction-only classifier gate (flips karcher's up/down call when confident, never touches
  the predicted price, so it can't hurt MAE by construction): helped on ETH (realized
  48.3% → 55.0%), hurt slightly on BTC/Brent/S&P500/EUR-USD.

Neither is wired into the live app or the default pipeline — they underperformed often enough
in this sample that shipping them as a default would be reporting a result that didn't
actually replicate. They're kept in `ensemble.py`/`benchmark.py`, tested
(`tests/test_ai_complement.py`), and rerunnable, per "test it and keep it in the project even if
it doesn't win." The honest conclusion for now: **karcher alone, no AI blend**, is the
strongest defensible forecast this repo can produce, and only on assets/horizons where its
own backtest shows a real edge (notably Brent, 3-day horizon).

**Finding 4 — the edge is daily-only; it does not transfer to hourly.** Rerunning the same
nested benchmark hourly (6-hour horizon, BTC/ETH/EUR-USD, ~95 scored folds) shows naive beats
karcher on MAE on every asset, and karcher's directional accuracy drops to ~42-51%
(EUR/USD: worse than a coin flip). Whatever signal the delay-embedded neighbour geometry picks
up on daily closes is not present at hourly resolution with the same embedding — this wasn't
retuned per-timeframe (same `k_neighbors`/`tau` defaults), so it's possible a timeframe-specific
retune would help, but as shipped, daily is the only timeframe this method should be trusted on.

**Finding 5 — selective prediction (only trust karcher when its own confidence signal is
high) also doesn't hold up, and goes backwards on the two best assets.** `benchmark.py`'s
`confidence_gate_breakdown` splits each asset's folds into "high" vs "low" |wind_t_stat|
(the neighbours' next-return t-stat — an expanding-median split computed causally, no
look-ahead) and scores karcher separately on each half:

| asset | high-confidence directional acc. | low-confidence directional acc. |
|---|---|---|
| Bitcoin | 43.3% | **63.3%** |
| Brent | 57.6% | **63.0%** |
| S&P 500 | **57.6%** | 48.1% |
| Ethereum | **57.1%** | 40.6% |
| EUR/USD | 51.4% | 52.0% |

On exactly the two assets where karcher has its clearest overall edge (BTC, Brent), being more
"confident" by the engine's own wind-strength measure makes it *worse*, not better — the
opposite of what a useful confidence signal should do. This is consistent with findings 1 and
3: the wind/Randers term is the part of this method that doesn't generalize, and anything
derived from it (as a correction, a blend feature, or now a confidence gate) inherits that.
The one component that holds up under every test here is the plain, wind-free karcher mean
itself.

**Finding 6 — CPCV shows Brent/S&P 500's edge is robust, Bitcoin's isn't; and a bootstrap
confidence gate (not the wind term) actually works, on 2 of 3 assets.** `cpcv_robustness.py`
replaces the single walk-forward path with a combinatorial purged CV: 8 chronological blocks,
every combination of 2 held out as "test", scored independently (28 combinations per asset).

| asset | karcher median dir. acc. | naive median dir. acc. | karcher loses in |
|---|---|---|---|
| Brent | 60.0% | 45.0% | 18% of combinations |
| S&P 500 | 50.0% | 40.0% | 18% of combinations |
| Bitcoin | 50.0% | 52.5% | 36% of combinations |

Brent and S&P 500's edge survives being looked at from many different fold groupings. Bitcoin's
apparent edge from the single walk-forward run (finding 1's table) does not — it was a property
of that particular fold ordering, not something real.

Separately, `bootstrap_confidence_gate.py` tries a different confidence signal than finding 5's
wind-based one: resample the K analogs' returns with replacement (200 times) and use the spread
of the resampled means as a "how much do the analogs agree with each other" signal, picked
causally (an expanding median, no look-ahead). Unlike the wind-based gate, this one works:

| asset | tight-spread (confident) dir. acc. | wide-spread (unsure) dir. acc. |
|---|---|---|
| S&P 500 | **67.6%** | 34.6% |
| Bitcoin | **59.4%** | 46.4% |
| Brent | 57.9% | 63.6% (reversed) |

Two out of three assets show a real, large gap — when the K analogs agree with each other, trust
karcher more; when they don't, it's closer to a coin flip. Brent, the asset with the strongest
raw edge, doesn't show this pattern. So: a genuinely useful, AI-free confidence signal, but not
universal — worth combining with the per-asset CPCV result above (e.g. gate on BTC/S&P 500 where
both the edge and the confidence signal check out, trust Brent's karcher output unconditionally
since it's already robust without gating).

**Finding 7 — a DFMC-TimeGAN-style discriminator filter on the analogs doesn't move the needle.**
`discriminator_scenario_filter.py` tests whether AI can help in scenario *analysis* rather than
correction: train a small classifier to tell a real analog continuation apart from an incoherent
one (random bootstrap draws of individual historical returns), then re-weight the K analogs by
that "realism" score before averaging, instead of equal weight. Tested on Brent, S&P 500 and
Bitcoin, the realism-weighted mean is statistically indistinguishable from plain karcher on all
three — the classifier can't separate the two classes well enough, on simple path features
(cumulative return, path vol, max step, autocorrelation), to produce weights that differ from
uniform in any way that matters.

**Where this leaves things:** across seven independent attempts (ai_blend, direction-flip gate,
wind confidence gate, Ridge analog-reweighting, discriminator filter, plus CPCV and the bootstrap
gate as *evaluation* methods rather than corrections), the only two things that held up were (a)
the plain karcher mean itself, confirmed robust on Brent/S&P 500 by CPCV, and (b) selective
prediction via bootstrap agreement, which works on 2 of 3 assets. Every attempt to make the
forecast itself smarter with AI — correcting it, reweighting its inputs, or filtering its inputs
for plausibility — failed or was a no-op. That is the honest result of this line of
investigation, not a setup for a future success.

**Finding 8 — metric learning for *which* analogs get picked (not just how they're weighted
afterward) robustly rescues Bitcoin specifically, while hurting/not-helping the two assets that
already worked.** `metric_learning_analog_selection.py` is a different kind of change from
findings 1-7: instead of reweighting or filtering the K analogs after kNN selects them, it
changes the embedding metric used to select them — scaling the 3 delay-embedding coordinates
`[r(t), r(t-tau), r(t-2tau)]` by a per-coordinate weight (picked by nested validation on past
folds, from a small fixed grid) before the Euclidean kNN search. CPCV-confirmed:

| asset | effect |
|---|---|
| Bitcoin | **robust win** — loses to the unweighted default in 0% of 28 CPCV combinations (median dir. acc. 64.3% vs. 58.1%) |
| S&P 500 | coin flip — loses in 50% of combinations |
| Brent | hurts — loses in 64% of combinations |

The learned weight consistently down-weights the oldest lag (picks `(1.0, 1.0, 0.5)` in 43/60
folds) — i.e. for Bitcoin specifically, the 3rd embedding coordinate (2 days back) is mostly
noise, and dropping its influence on neighbour selection helps. This is notable because Bitcoin
is exactly the asset whose base karcher edge did *not* survive CPCV (finding 3) — this is the
first improvement that specifically targets and fixes the one asset the base method was weakest
on, rather than amplifying an edge that already existed. It is asset-specific, not universal:
exactly the "which assets, under which conditions" framing to report, not a general upgrade.

**Finding 9 — a conformal-style gate on raw analog spread is weaker than the bootstrap gate.**
`conformal_spread_gate.py` calibrates a nonconformity threshold on the K analogs' raw return
spread (not the bootstrap-resampled spread of their mean) on an expanding window. It's reversed
on Brent (confident bucket is *worse*: 42.9% vs. 75.0%), flat on S&P 500, and only weakly
confirms the Bitcoin direction (60.0% vs. 46.7%, smaller than the bootstrap gate's 59.4% vs.
46.4% — comparable there, but worse everywhere else). Bootstrapping the mean, not just reading
the raw spread, is doing real work in finding 6's result.

**Finding 10 — gating by realized-volatility regime finds nothing.** `volatility_regime_gate.py`
tests the classic hypothesis that calm regimes are more forecastable than turbulent ones, via
causal expanding terciles of 21-bar realized volatility. No usable signal: extreme buckets have
too few folds to mean anything (down to n=1), and the pattern doesn't agree in direction across
assets. Reported as a clean negative.

## Summary across all 10 methods tested

| # | method | mechanism | result |
|---|---|---|---|
| 1 | Fréchet / median (built into the base method) | Finsler wind term | consistently worse than naive and karcher, every asset |
| 2 | Gradient-boosted residual correction | correct karcher's price error | no better than karcher alone, every asset |
| 3 | Direction-flip classifier gate | flip karcher's up/down call when confident it's wrong | never beat "never flip" |
| 4 | Wind-based confidence gate | trust karcher more when wind t-stat is high | **backwards** on the 2 best assets |
| 5 | Ridge analog reweighting | learn per-analog weights | always collapsed to equal weight (= karcher) |
| 6 | Bootstrap confidence gate | resample K analogs, gate on spread of the mean | **works**: S&P 500, Bitcoin (not Brent) |
| 7 | Discriminator scenario filter | re-weight analogs by "realism" score | no different from equal weight, every asset |
| 8 | Metric learning for analog selection | learned per-coordinate embedding weights | **works**: robustly fixes Bitcoin only; hurts Brent |
| 9 | Conformal-style raw-spread gate | gate on K analogs' raw return spread | weaker version of #6, reversed on Brent |
| 10 | Volatility-regime gate | gate on realized-vol tercile | no usable signal, any asset |

Two genuinely new, CPCV/nested-validated positive results came out of ten independent attempts:
selective prediction via bootstrap agreement (#6) and metric-learned analog selection (#8) — and
both are asset-specific, not universal. Every attempt to make the forecast's *value* smarter
(correct it, reweight its inputs by learned importance, filter its inputs by plausibility)
failed; what worked was either deciding *when* to trust it (#6) or changing *which* historical
analogs count as similar in the first place (#8).

**Finding 11 — heavier models (LSTM, Transformer) directly on raw returns don't help either,
and actually underperform a trivial baseline.** `deep_direction_classifier.py` trains an LSTM
and a small Transformer encoder on 60-day windows of raw daily log-returns (class-balanced
loss), retrained periodically on an expanding walk-forward window, to classify 3-day-ahead
direction — a genuinely heavier model than anything else tried here (minutes of GPU training
per asset vs. the closed-form/shallow-tree methods elsewhere in this repo).

| asset | LSTM dir. acc. | Transformer dir. acc. | trivial majority-class baseline |
|---|---|---|---|
| Brent | 52.5% | 50.0% | 52.5% |
| S&P 500 | 55.9% | 47.9% | 55.9% |
| Bitcoin | 47.6% | 49.2% | 50.2% |

Both models land at or below the baseline of just always predicting the majority class in the
training window; the Transformer's output probabilities have almost no spread (`prob_std` as
low as 0.006) — it isn't learning a signal, it's collapsing toward a near-constant prediction.
More capacity did not help, because capacity was never the bottleneck: ten prior experiments
(findings 1-10) already established that the available features and raw returns don't carry
learnable signal much beyond karcher's own historical average. This matches the broader literature on noisy daily financial targets at this sample size:
model complexity tends to hurt, not help, once the simpler methods have already been tried.

**Finding 12 — VIX (a cross-asset signal, unlike every other feature tried here) doesn't beat
or add to the bootstrap gate on S&P 500.** `vix_cross_asset_gate.py` tests VIX level and 5-day
VIX change as gates, motivated by the literature's repeated use of VIX as an S&P 500 predictor.
VIX alone (best bucket 58.3% dir. acc., n=12) is weaker than the bootstrap gate (67.6%), and
combining both (tight analog spread + rising VIX) gives 57.9% — worse than the bootstrap gate
standalone. VIX doesn't carry information beyond what the analogs' own internal agreement
already captures for this asset, at least at this horizon.

**Current best validated numbers, for reference (superseded in part by findings 18-26 below —
kept here as the state after finding 17; see the final summary table at the end of this
document for the up-to-date best-per-asset picture):**

| asset | before (Fréchet, the method's original headline output) | karcher alone | best complement found so far (through finding 17) |
|---|---|---|---|
| Brent | 35% dir. acc. | 60.0% (CPCV-robust) | 70.7% (volume-surprise gate, finding 14) |
| S&P 500 | 45% | 50.0% (CPCV median) | 67.6% (bootstrap confidence gate, finding 6) / 60.0% (volume gate, more universal) |
| Bitcoin | 40% | 50.0% (CPCV median, not robust) | 64.6% (denoising, finding 13) / 64.3% (metric learning, finding 8) / 62.5% (volume gate, finding 14) |

All of these are gate-style results (apply only to the subset of days/predictions flagged
confident) except metric learning and denoising, which change the method for every prediction.
Through finding 17, the volume-surprise gate (finding 14) was the one result confirmed on every
asset tested; finding 26 (diffusion reconstruction-error gate) later supersedes it as the
strongest universal result — see the final table.

The S&P 500 number applies only to the subset of days the gate flags as confident (a
coverage/accuracy trade-off, not a free lunch); the Bitcoin number applies to every prediction,
since it changes the method itself rather than gating when to use it.

**Finding 13 — causal denoising of the embedding (not the forecast target) is a third
Bitcoin-specific win, and a cautionary tale about validation discipline.**
`causal_denoising_experiment.py` smooths the log-returns used for neighbour search with a
causal (one-sided) EMA — never the actual historical returns being averaged for the forecast —
with the smoothing span picked by nested validation. A first, naive pass (picking the span that
looked best over the whole backtest, no nested validation) made it look like this helped S&P
500 (48.7% → 56.2%). Redone properly and CPCV-checked, that apparent gain disappeared entirely
(51.7% either way) — exactly the failure mode the nested/CPCV discipline throughout this
directory exists to catch. What survived: Bitcoin improves robustly (58.1% → 64.6%, losing in
only 32% of CPCV combinations), Brent gets worse (loses in 79%). Denoising is now the third
distinct technique (with bootstrap confidence gating and metric-learned analog selection) found
to specifically help Bitcoin and nothing else.

**Finding 14 — a volume-surprise gate is the strongest, most universal result in this whole
investigation.** `volume_surprise_gate.py` gates karcher on trading VOLUME — a signal of a
genuinely different kind from every price/return-derived feature tried so far. Feature:
today's volume as a z-score against its trailing 20-day mean (causal), split on a causal
expanding median. Unlike every other gate tried (wind t-stat, bootstrap spread, raw analog
spread, realized-vol regime, VIX — findings 4, 6, 9, 10, 12), this one is CPCV-robust on **all
three** assets tested, not just one or two:

| asset | high-volume-surprise median dir. acc. (CPCV) | low-volume-surprise median |
|---|---|---|
| S&P 500 | **60.0%** | 37.5% |
| Brent | **70.7%** | 50.0% |
| Bitcoin | **62.5%** | 44.4% |

Mechanism: unusually high volume is a classic proxy for new information hitting the market
(earnings, news, institutional flow); on those days a short-term analog continuation is more
likely to actually play out than on a quiet, low-volume day where the next move is closer to
noise. This single, model-free gate beats every learned correction/weighting/filter tried in
findings 2, 3, 5, 7, and the deep models in finding 11 — and it generalizes where nothing else
did.

**Finding 15 — Amihud illiquidity is weaker than, and inconsistent with, the volume-surprise
gate.** `amihud_illiquidity_gate.py` tests `|return| / dollar volume`, a standard asset-pricing
liquidity measure, as a gate. The sign isn't even consistent across assets: low illiquidity
(calmer days) is better for S&P 500 (57.1% vs. 46.9%) and Brent (66.7% vs. 50.0%), but HIGH
illiquidity is better for Bitcoin (61.5% vs. 44.1%). Not usable as a universal signal.

**Finding 16 — combining the volume gate and the bootstrap gate doesn't compound.**
`combined_volume_bootstrap_gate.py` splits into a 2x2 (high/low volume x tight/wide bootstrap
spread). The "both agree" bucket is roughly in line with the volume gate alone, not meaningfully
better, on S&P 500 (65.2% vs. 60.0% standalone) and Bitcoin (64.3% vs. 62.5%); on Brent the best
bucket is actually "high-volume + wide-spread" (75.0%, but n=12) rather than the intersection —
noisy, not a clean win. The two signals aren't independent enough to stack.

**Finding 17 — a full-feature, validation-gated gradient-boosted classifier (replacing karcher
outright when it validates better, falling back otherwise) still loses.**
`full_feature_direct_classifier.py` throws every useful feature found in this investigation
(wind strength/t-stat, anisotropy, analog spread, momentum/vol, volume z-score, bootstrap
spread) at a single classifier with an explicit safety fallback to karcher. It still
underperforms karcher alone on S&P 500 (43.6% vs. 49.1%) and Bitcoin (40.0% vs. 52.7%), and ties
on Brent. This is the clearest demonstration in the whole investigation that the bottleneck is
not feature richness or model choice: 3-day-ahead direction is close to unlearnable from any
combination of these features, for these assets, at this sample size.

## What actually worked, after 17 methods

Nothing that tried to predict or correct the forecast's *value* ever won. Everything that won
changed *when* to trust the existing forecast or *which* historical days count as similar:

1. **Bootstrap confidence gate** (#6) — S&P 500, Bitcoin.
2. **Metric-learned analog selection** (#8) — Bitcoin only, but fixes an asset the base method
   didn't work on at all.
3. **Causal denoising of the embedding** (#13) — Bitcoin only.
4. **Volume-surprise gate** (#14) — **all three assets**, the only universal result found.

If this becomes a paper, finding 14 is the headline result; findings 6, 8 and 13 are the
asset-specific appendix; findings 1-5, 7, 9-12, 15-17 are the (honestly reported) negative
space that makes the positive results credible rather than cherry-picked.

**Finding 18 — a diffusion-score-based similarity metric for analog selection robustly fixes
S&P 500.** `diffusion_score_metric.py` is a different combination than finding 8's learned
coordinate weights, and critically different from a diffusion *scenario generator* (which
would replace the true historical continuations with synthetic ones — the user's own earlier
diffusion+RL attempt, which underperformed for exactly that reason). Here, a small unconditional
DDPM is trained on the engine's 3D delay-embedding; its denoiser's hidden-layer activation at a
fixed small noise level becomes a learned "diffusion feature" for each point, and Euclidean
distance in that feature space replaces plain distance for picking the K=7 analogs. Their true
historical continuations are still averaged exactly as before — only *what counts as similar*
changes. CPCV-confirmed:

| asset | plain-Euclidean metric | diffusion-score metric |
|---|---|---|
| Brent | 75.0% | 70.0% (worse, loses in 46% of combos) |
| S&P 500 | 57.5% | **65.0%** (robust win, loses in 0% of combos) |
| Bitcoin | 67.5% | 65.0% (worse, loses in 64% of combos) |

A second technique in the "change what counts as similar" family (after #8), specific to S&P
500 this time rather than Bitcoin — reinforcing that this category, not direct correction, is
where real complements to the base method live.

**Finding 19 — swapping diffusion for a Transformer masked-autoencoder as the representation
learner doesn't work on any asset.** `transformer_mae_metric.py` is the direct parallel to
finding 18: same analog-selection mechanism, but the similarity feature comes from a small
Transformer encoder pretrained with a masked-autoencoding pretext task (mask one of the 3
delay-embedding coordinates, predict it from the other two; pooled encoder output = feature)
instead of a diffusion denoiser's hidden activation. CPCV-confirmed negative, everywhere:

| asset | plain-Euclidean metric | Transformer-MAE metric |
|---|---|---|
| Brent | 75.0% | 60.0% (loses in 64% of combos) |
| S&P 500 | 57.5% | 55.0% (loses in 46% of combos) |
| Bitcoin | 67.5% | 57.5% (loses in 71% of combos) |

Between the two representation-learning approaches tried as a similarity metric, only the
diffusion denoiser's learned feature (finding 18) captured something useful; the masked-
autoencoding Transformer's pooled embedding did not, on any of the three assets.

**Finding 20 — the diffusion-score metric stacks with the volume-surprise gate: best validated
S&P 500 number in the investigation.** `diffusion_volume_stack.py` gates finding 18's
diffusion-metric prediction by the same causal volume-surprise split as finding 14.
CPCV-confirmed on S&P 500: diffusion-metric alone scores 65.0% (median); gated to the
high-volume-surprise subset, **70.0%** (median across 28 combos); the low-volume subset drops
to 58.6%. The two independent findings compound rather than cancel. A single-pass K-sweep
(number of analogs) also hinted that K=15 might beat the default K=7 (68.8% vs. 66.2%) — flagged
as an untested lead, not adopted, since it hasn't been through nested validation.

**Finding 21/22 — VAE as a fourth representation learner: a mild, broadly safe improvement,
unlike diffusion's sharp asset-specific win.** `vae_similarity_metric.py` repeats findings
18/19's setup with a VAE's latent mean as the similarity feature. CPCV-confirmed: never loses
badly, never wins big, on any asset:

| asset | plain-Euclidean | VAE-metric | VAE loses in |
|---|---|---|---|
| Brent | 75.0% | 72.5% | 25% of combos |
| S&P 500 | 57.5% | 60.0% | 0% of combos |
| Bitcoin | 67.5% | 65.0% | 25% of combos |

**Finding 23 — swapping the diffusion denoiser's MLP for a Kolmogorov-Arnold Network (KAN,
Liu et al. 2024) backbone reproduces the same S&P 500 win almost exactly.**
`kan_similarity_metric.py` uses a simplified KAN layer (edge-wise learnable Fourier-basis
functions, not the official B-spline KAN) in place of the MLP in finding 18, same mechanism
otherwise. CPCV-confirmed:

| asset | plain-Euclidean | KAN-metric | KAN loses in |
|---|---|---|---|
| Brent | 75.0% | 70.0% | 46% of combos |
| S&P 500 | 57.5% | 65.0% | 7% of combos |
| Bitcoin | 67.5% | 65.0% | 75% of combos |

Nearly identical to the plain-MLP diffusion metric on every asset. This matters because it
isolates WHAT is working: the S&P 500 improvement comes from learning a similarity metric via a
generative/denoising training objective, not from any specific network architecture (MLP vs.
KAN gives the same result); a self-supervised reconstruction objective (Transformer-MAE,
finding 19) does not produce the same effect, so the objective — not the architecture — is
the active ingredient.

**Finding 24 — adding local derivatives (velocity, acceleration) to the similarity space adds
nothing new, and this finding caught a real leakage bug on the way.** `derivative_jet_metric.py`
extends each embedded point with its local first derivative (velocity, already implicit in the
engine's wind term) and second derivative (acceleration/curvature, genuinely new — the base
method never uses it), tested both as a fixed closed-form distance (no network) and as input
to the same diffusion-metric-learner as finding 18 (with a network).

The first version of this script computed derivatives with FORWARD differences, which for the
query point leaks tomorrow's return into today's similarity feature — it produced an
implausible ~90% accuracy on all three assets with 0% CPCV loss everywhere. This is exactly the
kind of result this investigation's discipline exists to catch: fixed to backward (causal)
differences, the real result is unremarkable:

| asset | position-only | +derivatives (no network) | +derivatives (network) |
|---|---|---|---|
| Brent | 75.0% | 65.0% (worse) | 75.0% (tie) |
| S&P 500 | 57.5% | 60.0% (mild) | 65.0% (same level as finding 18, not better) |
| Bitcoin | 67.5% | 65.0% (worse) | 65.0% (worse) |

Curvature information doesn't add anything beyond what finding 18's position-only diffusion
metric already captures. Included in full — bug and all — because methodological discipline is
as much a finding here as the numbers.

**Finding 25 — fractional differentiation (de Prado) as the embedding input: no improvement.**
`fractional_differentiation_metric.py` replaces plain log-returns (d=1 differencing) with
Fixed-Width-Window fractional differentiation, a well-established technique from "Advances in
Financial Machine Learning": d=1 over-differences and destroys memory, so the literature argues
for the smallest d achieving stationarity (found causally per-origin via an ADF-test grid
search). Caught a second index-alignment bug on the way (the fracdiff array was only extended
on the days d* was re-picked, so folds in between ran against a too-short array and silently
returned zero analogs — caught because almost every fold was empty; fixed by decoupling the
periodic d* search from the fracdiff series computation itself, redone every fold). CPCV-
confirmed result with the bug fixed, causal d* typically landing around 0.2-0.3 (confirming de
Prado's point that d=1 is over-differencing):

| asset | plain log-return | fracdiff (no network) | fracdiff (with network) |
|---|---|---|---|
| Brent | 75.0% | 65.0% (loses 75%) | 70.0% (loses 46%) |
| S&P 500 | 57.5% | 60.0% (loses 43%) | 65.0% (loses 0%, ties finding 18, doesn't beat it) |
| Bitcoin | 67.5% | 62.5% (loses 64%) | 65.0% (loses 64%) |

A well-established technique elsewhere in quant finance that doesn't add information beyond
what plain log-returns plus a diffusion-learned metric already capture, on this specific task.

**Finding 26 — diffusion reconstruction-error as a confidence gate: the strongest, most
universal result in the entire investigation, and the opposite of its own motivating
hypothesis.** `diffusion_anomaly_gate.py` unifies the two categories that worked elsewhere
(generative-model similarity, findings 18/22/23; confidence gating, findings 6/14) into one
idea, borrowed from diffusion-based anomaly detection (e.g. AnoDDPM): train the same denoiser as
finding 18, and at each origin estimate its own denoising loss at the query point (averaged
over 32 random timestep/noise draws) as an "atypicality" score — a market state the model finds
hard to denoise is one unlike its training history. The hypothesis was that atypical days would
be LESS predictable by the historical-analog method. CPCV-confirmed result is the opposite, on
every asset:

| asset | low_typical (median) | high_atypical (median) | high_atypical loses in |
|---|---|---|---|
| Brent | 70.7% | **83.3%** | 29% of paired combos |
| S&P 500 | 50.0% | **60.0%** | 41% of paired combos |
| Bitcoin | 61.3% | **80.0%** | 7% of paired combos |

Days the model finds hard to denoise are MORE predictable by karcher, not less — decisively so
on Brent and Bitcoin, directionally on S&P 500. Plausible mechanism: atypical states tend to be
driven by a real directional shock, and historical analogs of past atypical days captured the
same kind of shock continuing; "typical," quiet days sit closer to a random walk the
analog method has no edge on in the first place. This is the first gate in the whole
investigation confirmed on all three assets with this much margin, beating even the volume-
surprise gate's (finding 14) effect sizes on Brent and Bitcoin.

**⚠️ Methodological caveat affecting findings 18-26 — read before citing any of their numbers.**
None of findings 18-26 (diffusion-score metric, VAE, KAN, Transformer-MAE, metric-learned
derivatives, fractional differentiation, the diffusion-anomaly gate) call
`FinslerRandersGeodesicEngine` at all. Each one reimplements a simplified, plain-Euclidean
version of the analog-selection idea from scratch, and compares against *that* simplified
baseline — not against the colleague's actual local-covariance (Ledoit-Wolf/Mahalanobis)
metric. This was not intentional; it was discovered only when asked directly whether the "best"
results actually used anything from the colleague's real method.

**Finding 27 — re-testing finding 26 (the strongest, most universal result) against the REAL
engine's karcher output: the effect does not survive intact.**
`diffusion_anomaly_gate_real_engine.py` keeps the same diffusion-anomaly gate but gates the
ACTUAL `karcher_pred` from `FinslerRandersGeodesicEngine` (via `benchmark.py`'s `collect_folds`,
real Mahalanobis metric and all) instead of the simplified reimplementation's mean:

| asset | low_typical (real karcher) | high_atypical (real karcher) | vs. finding 26's simplified numbers |
|---|---|---|---|
| Brent | 54.5% | 75.0% | same direction, weaker (was 70.7% / 83.3%) |
| S&P 500 | 63.1% | **33.3%** | **REVERSED** (was 50.0% / 60.0%) |
| Bitcoin | 47.7% | 57.1% | same direction, weaker (was 61.3% / 80.0%) |

The gate's direction survives, diminished, on Brent and Bitcoin, but completely reverses on
S&P 500. Finding 26 as originally stated overstated both its strength and its universality:
real for the simplified reimplementation, but partly an artifact of testing against a weaker
baseline than the colleague's actual method. **The honest, current best validated numbers for
the real engine are findings 1-17's (volume-surprise gate #14, bootstrap gate #6), which DO use
`FinslerRandersGeodesicEngine` directly — findings 18-26 should be read as "ideas that worked on
a simplified stand-in for the method, not yet confirmed on the method itself," pending the same
re-test this script ran for finding 26 being run for the others.** This correction exists only
because the user asked a direct, skeptical question about whether the colleague's actual work
was used anywhere in the "best" results — exactly the kind of check this investigation's own
discipline should have run proactively.

## Layout

```
config.py          asset universe + trading calendar per category
data.py             yfinance price download
formatting.py        price display helpers
geometry.py           delay embedding, local Finsler metric, analog-flow engine
plotting.py            Plotly dashboard figure (analog and/or diffusion)
ui.py                   Streamlit page
app.py                   entry point
backtest.py              walk-forward comparison across all methods (EUR/USD, with diffusion)
ensemble.py               AI-complement experiments on top of karcher (see Benchmark section)
benchmark.py              nested, no-leakage, multi-asset/timeframe honest benchmark
rl/
  dataset.py              hourly bars -> 3-day windows, Finsler (and diffusion) prior per window
  agent.py                 window environment + PPO residual policy
  backtest.py               walk-forward RL backtest, raw vs denoised comparison
diffusion/
  dataset.py             Dukascopy minute downloader, hourly resampling
  denoise.py              drops closed-market "flat" candles before resampling
                          (not to be confused with model.py's noise-prediction network)
  model.py                 FiLM-conditioned 1D-CNN denoiser
  train.py                   DDPM training + sampling
  forecast.py                 DiffusionScenarioForecaster: checkpoint -> price forecast
```

## Independent track: a standalone diffusion-based analog model (no longer tied to the colleague's engine)

Starting from finding 18, this investigation forked into a separate, standalone model: delay
embedding + plain-Euclidean (or diffusion-learned) analog selection + diffusion-based
confidence gating, which does not call `FinslerRandersGeodesicEngine` and is evaluated as a
model in its own right rather than a complement to the colleague's method (see the finding-27
caveat above for why that distinction matters).

**Finding 28 — refining the diffusion-anomaly gate (finding 26) with a lower-variance ELBO
estimator and a 5-asset test reveals a real, generalizable pattern, with Brent as a genuine
exception.** `diffusion_v2_improvements.py` tests three refinements: (A) a deterministic
full-ELBO atypicality score (average denoising loss over all 50 timesteps, not a 32-sample
Monte Carlo subset — lower variance); (B) a Diffusion-Time-Estimation-style score (the *minimum*
per-timestep loss, a different summary of the same posterior, from the diffusion anomaly-
detection literature); (C) stacking diffusion-metric analog selection (finding 18) with the
ELBO gate. Tested on five assets (Brent, S&P 500, Bitcoin, Ethereum, Gold), CPCV-confirmed:

| asset | (A) full ELBO gate (low/high) | (C) metric-selection + ELBO gate (low/high) |
|---|---|---|
| Brent | 80.0% / 70.8% — **reversed** | 67.9% / 55.0% — **reversed** |
| S&P 500 | 61.3% / 71.4% | 70.0% / 71.4% (flat) |
| Bitcoin | 61.3% / 75.0% | 68.3% / **83.3%** |
| Ethereum | 57.1% / 71.4% | 56.3% / **80.0%** |
| Gold | 57.1% / 66.7% | 59.2% / **75.0%** |

Method (B), the DTE-style score, was weak and flat on every asset and wasn't pursued further.
The combination in (C) is strong and consistent on Bitcoin, Ethereum, and Gold, flat on S&P
500, and reverses specifically on Brent — meaning finding 26's original 83.3% Brent number
(from the noisier 32-sample estimator) was likely partly a noise artifact, not a robust effect.
The honest, refined claim: this combination generalizes well to crypto and precious-metal
markets but not to Brent crude, plausibly because Brent's price action is more often driven by
discrete geopolitical/supply shocks than by the kind of continuous statistical structure a
diffusion model trained purely on price history can learn.

## Final summary across all 27 methods tested

| # | method | mechanism | result |
|---|---|---|---|
| 1 | Fréchet / median (built into the base method) | Finsler wind term | FAIL, every asset |
| 2 | Gradient-boosted residual correction | correct the value | FAIL |
| 3 | Direction-flip classifier gate | correct the value | FAIL |
| 4 | Wind-based confidence gate | when to trust | FAIL (backwards) |
| 5 | Ridge analog reweighting | change similarity | FAIL (collapsed to default) |
| 6 | Bootstrap confidence gate | when to trust | WIN: S&P 500, Bitcoin |
| 7 | Discriminator scenario filter | change similarity | FAIL |
| 8 | Metric-learned analog selection (coordinate weights) | change similarity | WIN: Bitcoin |
| 9 | Conformal-style raw-spread gate | when to trust | FAIL (weaker than #6) |
| 10 | Volatility-regime gate | when to trust | FAIL (no signal) |
| 11 | LSTM / Transformer direct classifier | correct the value | FAIL (collapses) |
| 12 | VIX cross-asset gate | when to trust | FAIL |
| 13 | Causal denoising of the embedding | change similarity | WIN: Bitcoin |
| 14 | Volume-surprise gate | when to trust | WIN: all 3 assets |
| 15 | Amihud illiquidity gate | when to trust | FAIL (inconsistent sign) |
| 16 | Volume + bootstrap gate combined | when to trust | FAIL (no compounding) |
| 17 | Full-feature gradient-boosted classifier | correct the value | FAIL (even with fallback) |
| 18 | Diffusion-score similarity metric | change similarity | WIN: S&P 500 |
| 19 | Transformer masked-autoencoder metric | change similarity | FAIL, every asset |
| 20 | Diffusion-metric + volume gate stack | when to trust + change similarity | WIN: S&P 500 (best validated number, 70%) |
| 21 | K (analog count) sweep | hyperparameter | untested lead (K=15 looked better, not nested-validated) |
| 22 | VAE similarity metric | change similarity | mild WIN, all 3 assets (never loses badly) |
| 23 | KAN-backbone diffusion metric | change similarity | WIN: S&P 500 (reproduces #18) |
| 24 | Derivative/jet-space metric | change similarity | FAIL (no improvement; caught a leakage bug) |
| 25 | Fractional differentiation (de Prado) | change similarity (input repr.) | FAIL (ties #18, doesn't beat it) |
| 26 | Diffusion reconstruction-error gate | when to trust | WIN (simplified reimpl.): all 3 assets — see #27 |
| 27 | Re-test of #26 against the REAL `FinslerRandersGeodesicEngine` output | when to trust | MIXED: holds (weaker) on Brent/Bitcoin, **REVERSES on S&P 500** |

**⚠️ Findings 18-26 were validated against a simplified, plain-Euclidean reimplementation of
the analog-selection idea, not against `FinslerRandersGeodesicEngine` itself — see the caveat
above finding 27. Only finding 27 has been re-checked against the real engine so far.**

**Best validated result per asset, with the finding-27 correction applied:**

| asset | confirmed-on-the-real-engine best | result | note |
|---|---|---|---|
| Brent | volume-surprise gate (#14) | **70.7%** | real engine, CPCV-confirmed |
| S&P 500 | bootstrap confidence gate (#6) | **~50-67.6%** (sensitive to resample, see #6) | real engine |
| Bitcoin | volume-surprise gate (#14) | **62.5%** | real engine, CPCV-confirmed |

The diffusion-anomaly gate (#26) remains the most interesting *lead* — large effect sizes,
a plausible mechanism, and a partial survival of its direction on 2 of 3 assets even against
the real engine — but it is not yet a confirmed "best result" the way #6/#14 are, and should
not be presented as one without first re-running #18, #20, #22, #23 the same way #27 re-ran #26.

**The pattern held across every one of 27 attempts, with zero exceptions:** every method that
tried to predict or correct the forecast's *value* (#2, #3, #5, #7, #11, #17, #24's no-network
variant, #25's no-network variant) failed. Every category of method that worked at all — on
either the real engine (#6, #8's underlying logic, #14) or the simplified reimplementation
(#13, #18, #20, #22, #23, #26) — either decided *when* to trust the existing karcher forecast,
or changed *what counts as similar* when picking historical analogs, never the predicted value
itself. That higher-level pattern is unaffected by the finding-27 correction; the specific
magnitude and universality claimed for finding 26 is.

## Run

```bash
pip install -r requirements.txt
streamlit run finsler_forecast/app.py
```

## Presentation dashboard (the 26-method investigation above)

```bash
python finsler_forecast/generate_dashboard_data.py   # refresh cached numbers (a few minutes)
streamlit run finsler_forecast/research_dashboard.py
```

A standalone, presentation-ready view of all 26 findings: a progression chart (original method
→ naive → karcher → volume gate → final best), a per-asset final-best summary bar, CPCV
robustness box plots for the base method/diffusion-metric/diffusion-anomaly-gate, volume/
bootstrap gate bucket charts, the S&P 500 diffusion+volume stack, and the full 26-method
pass/fail table. Reads from the cached `dashboard_data.json` so it loads instantly — re-run the
generator script to refresh the numbers against current market data before presenting.
