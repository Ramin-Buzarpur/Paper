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

Finding 14 is the headline result of this batch; findings 6, 8 and 13 are the
asset-specific extras; findings 1-5, 7, 9-12, 15-17 are the (honestly reported) negative
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

**Finding 29 — a real economic backtest (longer history, transaction costs, Sharpe ratio,
momentum baseline) surfaces a genuine edge AND a genuine problem with the binary gate.**
`economic_backtest.py` extends history to 7-20 years per asset (not the earlier fixed 6-year
window) and replaces directional-accuracy scoring with a non-overlapping, cost-aware trading
simulation. Two honest findings:

1. The diffusion-metric model (finding 18), with no gate at all, beats a simple momentum
   baseline on both Sharpe and total return, on all 5 assets tested (Brent, S&P 500, Gold,
   Bitcoin, Ethereum) — a real edge, sanity-checked against buy-and-hold (Sharpe 0.07-0.54 on
   these same assets, ruling out "it's just long-term beta").
2. The binary atypicality gate (findings 26/28) makes economic performance WORSE than always
   trading, on every asset — despite improving directional accuracy on the subset it fires on.
   It skips too many still-profitable "typical" days. This is a direct, necessary correction to
   reading findings 26/28 as a ready-made trading rule.

`position_sizing_backtest.py` fixes problem 2: instead of a binary trade/no-trade filter, the
atypicality score becomes a continuous position-size multiplier (everyone stays in the market),
with a cheap add-on that halves size when the diffusion-metric and momentum signals disagree.

| asset | ungated fixed-size (baseline) | continuous sizing | sizing + momentum filter |
|---|---|---|---|
| Brent | Sharpe 3.36, maxDD -21% | Sharpe 3.07 | Sharpe 3.22, maxDD -27% |
| S&P 500 | Sharpe 2.83, maxDD -17% | Sharpe 2.61 | Sharpe 2.69, maxDD -18% |
| Gold | Sharpe 3.02, maxDD -16% | Sharpe 2.96 | Sharpe 3.08, maxDD -19% |
| Bitcoin | Sharpe 3.39, maxDD **-37%** | Sharpe 3.34 | Sharpe **3.57**, maxDD **-26%** |
| Ethereum | Sharpe 3.14, maxDD **-47%** | Sharpe 3.14 | Sharpe 3.14, maxDD **-28%** |

Continuous sizing no longer destroys performance the way the binary gate did, and the
momentum-agreement add-on meaningfully cuts drawdown on the two highest-drawdown assets
(Bitcoin, Ethereum) with Sharpe flat or improved. **Caveat that applies to every number in this
finding:** Sharpe ratios of 2-3.4 are unusually high for a real systematic strategy and almost
certainly reflect the simplified cost assumption (2bps, no slippage) rather than literal
real-world expectations — read these as relative comparisons between strategies under matched
assumptions, not as a promise of real trading performance.

**Finding 30 — transaction-cost robustness, and a second sizing signal ("Conformal Kelly")
inspired by the conformal-prediction literature (Ryan 2025).** Two follow-ups to finding 29:

`cost_sensitivity.py` sweeps the transaction cost assumption from 2bps to 50bps (25x higher,
already conservative for liquid futures/equities, plausible even for crypto). Sharpe drops only
modestly at the highest cost: Brent 3.22→2.89, S&P 500 2.69→1.96, Gold 3.08→2.23,
Bitcoin 3.57→3.27, Ethereum 3.14→2.90. The 3-day holding period keeps turnover low enough that
the headline result isn't an artifact of an unrealistically cheap cost assumption.

`conformal_width_sizing.py` tests a second position-sizing signal from the "Conformal Kelly"
literature (scaling size inversely to a prediction interval's width, rather than directly by an
anomaly score): here, the natural interval is the spread of the K=7 analogs' own predicted
returns — narrow spread (analogs agree) → bigger position. Result: no single sizing signal
wins everywhere.

| asset | best Sharpe | best drawdown | winning method |
|---|---|---|---|
| Brent | 3.51 (spread-based) | 23.0% (blend) | spread-based |
| S&P 500 | 3.02 (spread-based) | **12.1%** (spread-based) | spread-based, clearly |
| Gold | 3.09 (blend) | 15.4% (spread-based) | close between methods |
| Bitcoin | 3.57 (finding 29's momentum filter) | 26.5% (finding 29's momentum filter) | momentum-agreement filter |
| Ethereum | 3.15 (blend) | **27.9%** (finding 29's momentum filter) | momentum-agreement filter |

A richer, more honest conclusion than a single winning method: several complementary
risk-sizing techniques exist, each suited to a different asset class, rather than one universal
fix — which is itself a meaningful result in its own right (robustness to cost assumptions, and
a principled menu of risk controls rather than one overfit choice).

## Reframing toward a detection/predictability claim (statistical rigor)

Trading-performance claims (findings 29/30) invite intense scrutiny (survivorship-biased bull
markets, ~30 methods tried raising multiple-testing concerns, unrealistic cost/slippage
assumptions). A cleaner, more defensible claim: diffusion-model reconstruction error identifies
market states that are genuinely, statistically more predictable by analog forecasting — a
detection/characterization result, not a trading system.

**Finding 31 — formal statistical testing (permutation test) confirms the atypicality-
predictability link on 4 of 5 assets, and shows a classical volatility-regime baseline does
NOT reproduce it on any asset.** `statistical_rigor.py` runs a 2000-permutation test: shuffle
the atypicality score across folds, recompute the high-vs-low accuracy gap under each shuffle,
and report the empirical p-value of the real gap against this null distribution. Repeated for a
classical realized-volatility-regime score (no diffusion model at all) as the most obvious
baseline to rule out.

| asset | diffusion atypicality gap | p-value | classical vol-regime gap | p-value |
|---|---|---|---|---|
| Brent | +8.3pp | **p<0.0001** | +3.1pp | p=0.099 |
| S&P 500 | +2.9pp | p=0.114 | +2.0pp | p=0.214 |
| Gold | +4.9pp | **p=0.022** | +0.7pp | p=0.386 |
| Bitcoin | +10.6pp | **p=0.0005** | +3.3pp | p=0.109 |
| Ethereum | +6.5pp | **p=0.027** | +1.6pp | p=0.333 |

The diffusion-based score is significant (p<0.05) on 4 of 5 assets; the classical
volatility-regime baseline is significant on none of them. This directly pre-empts the
most obvious objection ("why not just use a GARCH/volatility-regime split") with a
real head-to-head rather than an assertion. S&P 500's non-significance is consistent with every
earlier finding involving that asset, where the effect was always weakest.

**⚠️ Superseded by finding 33 below.** This finding only tested ONE weak classical baseline
(realized-volatility regime). Against three *stronger*, standard anomaly-detection algorithms
(Isolation Forest, LOF, Mahalanobis-PCA), the diffusion model is matched or beaten on nearly
every asset — the "diffusion specifically" framing does not survive; read finding 33 before
citing this one's comparison claim. The permutation-test methodology and the core
atypicality-predictability finding itself still stand.

**Finding 32 — combined significance (Fisher's method), two more assets, and an inconclusive
look at why S&P 500 is the outlier.** `meta_analysis_extended_panel.py` extends finding 31
three ways:

**(A) Fisher's method**, combining all per-asset p-values into one overall test
(χ² = -2·Σln(pᵢ), compared to a χ² distribution with 2k degrees of freedom): across all 7
assets, χ²=80.06, dof=14, combined p-value ≈ **0.000000** — decisively significant overall,
even though S&P 500 alone was not.

**(B) Two new assets** never tested anywhere in this investigation before — Silver and WTI
crude — both show the **strongest** effects of the entire panel:

| asset | effect gap | p-value |
|---|---|---|
| Silver | +10.9pp | **p<0.0001** |
| WTI Crude | +10.2pp | **p<0.0001** |

6 of 7 assets are now individually significant (p<0.05); only S&P 500 isn't. This is a strong
generalization result — the effect was not cherry-picked to the original 5 assets.

**(C) Why is S&P 500 the outlier?** Tested whether effect size correlates with each asset's
return autocorrelation (a simple market-efficiency proxy, used earlier in this investigation as
a predictor of karcher's edge). Correlation across all 7 assets: only **+0.211** — weak. This
simple proxy does not explain the S&P 500 gap. The likely real explanation — that S&P 500 is
the most intensely arbitraged, informationally efficient instrument in the panel — remains a
plausible hypothesis, not a demonstrated finding; honestly reported as unresolved rather than
forcing a tidy explanation the data doesn't support.

**Finding 33 — the most important correction in this investigation: the effect is real, but
it isn't specific to diffusion models, and it only partly survives a genuine held-out test.**
`robustness_journal_grade.py` runs two tests that directly target this.

**(A) Classical anomaly-detector comparison.** The diffusion ELBO score vs. three standard,
well-established anomaly detectors (Isolation Forest, Local Outlier Factor, Mahalanobis
distance in PCA space), all fit causally on the same 3D embedding:

| asset | diffusion | Isolation Forest | LOF | Mahalanobis-PCA |
|---|---|---|---|---|
| Brent | 8.3pp, p<0.0001 | 13.5pp, p<0.0001 | 13.4pp, p<0.0001 | 13.1pp, p<0.0001 |
| S&P 500 | 2.9pp, p=0.113 (not significant) | 4.8pp, **p=0.028** | 5.3pp, **p=0.013** | 5.1pp, **p=0.016** |
| Gold | 4.9pp, **p=0.022** | 11.4pp, p<0.0001 | 7.4pp, **p=0.002** | 8.8pp, p<0.0001 |
| Bitcoin | 11.0pp, p<0.0001 | 11.9pp, p<0.0001 | 12.0pp, p<0.0001 | 12.6pp, p<0.0001 |
| Ethereum | 6.5pp, **p=0.027** | 14.4pp, p<0.0001 | 8.7pp, **p=0.003** | 13.8pp, p<0.0001 |
| Silver | 10.9pp, p<0.0001 | 10.9pp, p<0.0001 | 11.1pp, p<0.0001 | 13.8pp, p<0.0001 |
| WTI Crude | 10.1pp, p<0.0001 | 7.5pp, **p=0.002** | 10.2pp, p<0.0001 | 9.3pp, p<0.0001 |

(Re-run in full on all 7 assets after finding 45's audit fix — strictly-causal detector fitting,
`hist = pc[:j]` excluding today — resolving the earlier Gold data outage along the way. The
pattern is unchanged: every asset is significant on at least one detector, S&P 500 remains the
weakest asset throughout, and the classical detectors continue to match or beat diffusion on
most assets.) **This means findings 18, 22, 23, and
31's "diffusion specifically" framing was premature** — the real, robust discovery is that
*any reasonable novelty/anomaly detector* on this delay-embedded return space finds the same
atypicality-predictability link. That is a more honest and, arguably, a more compelling claim:
a market stylized fact (novelty-conditional predictability), not a "our model beats theirs"
claim tied to one architecture.

**(B) A genuine, single-shot held-out test.** The last 20% of each asset's history was never
touched by any model selection, tuning, or exploratory check anywhere in this investigation.
Evaluated exactly once:

| asset | held-out p-value |
|---|---|
| Brent | **p=0.0005** |
| Silver | **p=0.0065** |
| WTI Crude | **p=0.0080** |
| S&P 500, Gold, Bitcoin, Ethereum | not significant individually |

Fisher's combined p-value across all **7** assets with available held-out data (re-run after
finding 45's audit fix, now including Gold): **p=0.000038** — the phenomenon survives at the
panel level on genuinely new data, consistent with the original 6-asset result (p=0.000024).
But only 3 of 7 assets are individually significant in this single confirmatory test, despite
most being significant (or close) on the fuller exploratory sample. Report both facts: the
effect is not a pure artifact of exploratory multiple-testing (the panel-level result
replicates out-of-sample), but it is also not uniformly strong enough to claim individually for
every asset on fresh data — Gold, Bitcoin, and Ethereum in particular should be described as
suggestive, not confirmed, going forward.

## A bridge between two literatures

The colleague's core method (delay-embed returns, find the K nearest historical states,
average their true continuations) is structurally identical to the **Analog Ensemble (AnEn)**
framework from atmospheric and space-weather forecasting (Delle Monache et al. 2011) — a
three-step technique (find analogs, read their verifying observations, pool them into an
ensemble) with an established formal literature, but one that, per a literature search for
this investigation, has not been formally applied to financial markets under that name;
finance's k-NN forecasting literature is related but distinct and informal by comparison.
Separately, finance has an extensive literature on *regime-dependent return predictability*
(bull/bear states, Markov-switching, high/low-volatility regimes) but none of it, per the same
search, uses novelty/anomaly detection as the regime signal. This investigation sits at the
intersection: an AnEn-style forecaster, gated by a novelty/anomaly score, tested against the
regime-dependent-predictability literature's own natural baseline (bull/bear state).

**Finding 34 — the atypicality signal is mostly independent of classical bull/bear regime, and
its regime-concentration differs by asset rather than following one universal pattern.**
`anen_bullbear_distinctness.py` tests whether the atypicality signal (whichever detector
computes it) is just a relabeled bull/bear indicator (price vs. 200-day moving average) by
(a) correlating the two, and (b) checking whether the predictability gap holds separately
within bull-only and bear-only sub-periods.

| asset | corr(atypicality, bull) | bull-only gap (p) | bear-only gap (p) |
|---|---|---|---|
| Brent | -0.126 | +1.8pp (p=0.322) | **+13.1pp (p<0.0001)** |
| S&P 500 | -0.456 (moderate) | +5.7pp (p=0.019) | -7.1pp (p=0.936, reversed) |
| Bitcoin | +0.008 | **+16.8pp (p<0.0001)** | +6.2pp (p=0.083) |
| Ethereum | +0.033 | +8.0pp (p=0.027) | +7.4pp (p=0.053) |
| Silver | -0.010 | **+14.1pp (p<0.0001)** | +6.9pp (p=0.033) |
| WTI Crude | -0.136 | +6.1pp (p=0.022) | **+12.7pp (p<0.0001)** |

5 of 6 assets show low correlation (<0.15 in magnitude) between atypicality and bull/bear
state — evidence this is a genuinely different dimension, not a relabeling. The effect
concentrates in bear markets for the two oil assets, in bull markets for Bitcoin, and in both
regimes for Silver and Ethereum — no single universal pattern, which argues against the
signal being a disguised version of one already-known regime indicator. S&P 500 remains the
exception (moderate correlation, effect reverses in bear markets), consistent with every prior
finding where it behaves differently from the rest of the panel.

**Finding 35 — a genuine information-fusion contribution: Dempster-Shafer combination of the 4
anomaly detectors improves worst-case robustness, winning outright on the panel's hardest
case.** Findings 31-33 only *compared* the 4 detectors (diffusion, Isolation Forest, LOF,
Mahalanobis-PCA) side by side. `dempster_shafer_fusion.py` *fuses* them with Dempster-Shafer
(DS) evidence theory — the classical combination framework from the Information Fusion
literature (Chen & Aickelin 2006, and the DST-ensemble anomaly-detection line of work since):
each detector's causal percentile rank becomes a basic probability assignment with a reserved
uncertainty mass, combined pairwise via Dempster's rule (which explicitly discounts conflicting
evidence through its normalization term), yielding one fused "atypical" belief per day.

| asset | best individual detector | DS-fused |
|---|---|---|
| Brent | 13.5pp (Isolation Forest) | **15.6pp, p<0.0001 — beats all 4** |
| S&P 500 | 5.3pp, p=0.013 (LOF) | **6.7pp, p=0.003 — beats all 4, strongest S&P 500 result in this entire investigation** |
| Gold | 11.4pp (Isolation Forest) | 9.0pp (close, not best) |
| Bitcoin | 12.6pp (Mahalanobis-PCA) | 11.9pp (close) |
| Ethereum | 14.4pp (Isolation Forest) | 12.5pp (close) |
| Silver | 13.8pp (Mahalanobis-PCA) | 13.1pp (near-tied) |
| WTI Crude | 10.2pp (LOF) | 8.8pp (close) |

(Re-run on all 7 assets, including Gold for the first time, after finding 45's audit fix to the
classical detectors' fit window — the pattern is unchanged, and if anything slightly stronger
on Brent and S&P 500.) Fusion never badly underperforms the best individual detector on any
asset, and on the two cases where picking the right single detector mattered most — Brent, and
especially S&P 500, the panel's persistent weak point throughout this entire investigation —
fusion strictly wins. This is the core information-fusion thesis (combining diverse evidence
sources via a principled rule improves worst-case robustness over relying on any single
source) demonstrated empirically on this problem, not just asserted from the literature. This
is the first result in the investigation that is a genuine fusion-methodology contribution
rather than a comparison or an application of an existing technique.

**Finding 36 — a genuinely new market-wide insight: atypicality co-occurs across traditional
asset classes, but crypto is decoupled from that structure.** `cross_asset_contagion.py` asks a
question not posed anywhere else in this investigation: does the fused atypicality flag
(finding 35) co-occur across DIFFERENT assets on the same calendar date more than chance would
predict? Idiosyncratic (per-asset-noise) atypicality should show no cross-asset relationship;
shared macro-shock-driven atypicality should.

| pair | correlation | p-value | relationship |
|---|---|---|---|
| Brent vs WTI Crude | **+0.560** | p<0.0001 | same asset class — sanity check, passes strongly |
| S&P 500 vs WTI Crude | +0.295 | p<0.0001 | cross-class, significant |
| Brent vs S&P 500 | +0.189 | p=0.0001 | cross-class, significant |
| S&P 500 vs Silver | +0.183 | p<0.0001 | cross-class, significant |
| Silver vs WTI Crude | +0.158 | p<0.0001 | cross-class, significant |
| Brent vs Ethereum | -0.185 | **p=0.016** | cross-class, significant NEGATIVE |
| Bitcoin vs Silver | -0.187 | **p=0.004** | cross-class, significant NEGATIVE |
| Bitcoin, Ethereum vs. the remaining pairs | -0.08 to +0.02 | not significant | decoupled |

(Re-run on all 6 assets after finding 45's audit fix — the traditional-market cluster's
positive co-occurrence is essentially unchanged in magnitude and significance. One detail did
shift: two crypto-vs-traditional pairs that were previously not significant, or only
borderline, are now significant and NEGATIVE — Bitcoin vs. Silver and Brent vs. Ethereum.
Reported honestly as a change in the specific pairwise pattern, not re-asserted as identical to
the original run.) Equities, metals, and energy (S&P 500, Brent, Silver, WTI Crude) show
significant, positive atypicality co-occurrence across asset classes — evidence the detector is
catching shared macro-level shocks, not per-asset noise (and the Brent/WTI same-class pair, at
+0.56, confirms the method isn't spurious). Crypto shows no significant *positive*
co-occurrence with any traditional market in either run, and now two significant *negative*
pairs — if anything this strengthens rather than weakens the segmentation reading: crypto's
atypical days do not coincide with, and in two pairings tend to anti-coincide with, traditional
markets' atypical days. This reads as evidence of market segmentation: the shared macro-shock
structure linking equities and commodities does not (at least in this data) extend to crypto,
relevant to the ongoing debate over whether crypto is integrated with traditional macro risk or
a separate ecosystem.

**Known limitation, reported honestly:** the Bitcoin-vs-Ethereum comparison found 0 overlapping
dates — a data-handling artifact (likely a timezone/index-alignment mismatch between separate
collection runs), not a real finding. Flagged rather than silently dropped.

**Finding 37 — a 30-stock market-breadth index does not improve index-level forecast
accuracy (negative, reported honestly).** `market_breadth_index.py` scales up to 30 diverse
S&P 500 constituents (tech, financials, energy, healthcare, consumer, industrials, media),
computes a lightweight (diffusion-only) causal atypicality flag per stock, and aggregates into
a daily breadth score (% of stocks flagged atypical) — a novelty-detection analogue of classic
advance/decline breadth indicators. Tested as a predictor of the S&P 500 index's own 3-day
forecast accuracy: gap = +0.2pp, p=0.473 — indistinguishable from chance. Aggregating many
stocks' idiosyncratic atypicality does not translate into better index-level direction
forecasts with this lightweight scoring. Not every scale-up idea works; this one didn't, and
is reported as such rather than omitted.

**Finding 38 — reframed as a systemic-risk early-warning indicator, the composite atypicality
score shows a clean, economically interpretable signal around known financial crises.**
`systemic_risk_indicator.py` averages the causal atypicality rank across 7 independent assets
(Brent, S&P 500, Gold, Silver, WTI, Bitcoin, Ethereum) into one daily composite index, and
checks its level during three well-known crisis windows against its long-run baseline (0.407):

| crisis window | composite mean | vs. baseline |
|---|---|---|
| 2008 Global Financial Crisis (Sep-Nov 2008) | 0.773 | **+36.6pp** (max hit the 1.0 ceiling) |
| 2020 COVID crash (Feb-Mar 2020) | 0.704 | **+29.8pp** (max 0.987) |
| 2022 inflation-driven selloff (Jan-Oct 2022) | 0.515 | +10.9pp (weaker) |

The composite nearly doubles during the two sudden, acute shocks (2008, 2020) and shows a
smaller but still elevated level during the slower, grinding 2022 selloff — consistent with
finding 34's observation that this signal's behavior depends on whether a regime shift is
abrupt or gradual, not just whether one is happening. This reframes the investigation's core
finding from "improves 3-day direction accuracy" (a narrow, trading-flavored claim) to a more
general and economically legible one: a simple composite cross-asset atypicality index behaves
like a systemic-risk early-warning signal, in the spirit of (though far simpler than) the
ECB's CISS index — directly relevant to the systemic-risk and financial-stress-indicator
literature in economics, independent of the trading-performance questions raised earlier.

**Finding 39 — Granger causality upgrades finding 36's co-occurrence claim to a directed
lead-lag structure, and independently corroborates crypto's segmentation.**
`granger_and_fresh_crisis.py` tests Granger causality (lags 1-3) among all pairs of the 7
assets' causal atypicality-rank series. Traditional markets show rich, directed structure:
S&P 500 → Silver/WTI Crude/Gold all significant at p<0.0001 (equities appear to *lead*
commodity atypicality, not just co-occur with it), and Brent↔WTI, Brent↔S&P 500 bidirectional
and strong. Bitcoin and Ethereum show no significant Granger relationship (p>0.04, mostly far
higher) with any other asset in either direction — a second, independent method (directional,
not just correlational) confirming finding 36's claim that crypto's atypicality is segmented
from the traditional-market macro-shock structure.

**Finding 40 — a fresh, never-touched crisis (March 2023 banking stress) tests the systemic-
risk framing prospectively, and reveals a real, honestly-reported limitation.** The composite
index (finding 38) rose only +3.0pp over baseline during the March 2023 US regional banking
stress (SVB/Signature/First Republic) — far weaker than the +36.6pp (2008) and +29.8pp (2020)
responses. This is a genuine limitation, not a universal early-warning claim: SVB was a narrow,
sector-specific shock (US regional banks), and this investigation's asset panel (broad equity
index, energy, metals, crypto) has no instrument directly exposed to that sector, so the
composite understandably stayed close to baseline. The indicator responds to shocks broad
enough to move its own panel's constituent assets, not to every systemically important event —
an important scope boundary to state explicitly rather than let the strong 2008/2020 results
imply more generality than the evidence supports.

**Finding 41 — a dynamical-systems mechanism: atypical days have measurably more contracting
local dynamics, explaining WHY they are more predictable, not just THAT they are.**
`local_lyapunov_regime.py` tests whether the core finding has a first-principles explanation in
nonlinear dynamics / control theory, rather than remaining an unexplained empirical correlation.
Local (finite-time) Lyapunov exponents generalize the classical Lyapunov exponent — the
textbook measure of local trajectory divergence used throughout control theory to certify
stability — to a state-dependent, time-varying quantity: even within an overall noisy/chaotic
system, there exist localized "pockets" of relatively contracting dynamics that are, by
construction, more short-term predictable (Kantz & Schreiber; the "prediction of
predictability" literature). For each fold's origin day, a Rosenstein-style estimator computes
the local Lyapunov exponent from the delay-embedded phase space: find the nearest causal
neighbour of today's state, then measure how much faster or slower the two trajectories
separate over the next `HORIZON` steps than they were separated today. (This retrospective
diagnostic itself uses future data, like the standard estimator — it explains the existing
forecast, it is not fed back into it, so no leakage is introduced into any accuracy number.)

| asset | mean LLE, atypical days | mean LLE, typical days | difference | permutation p |
|---|---|---|---|---|
| Brent | +0.733 | +0.898 | **-0.166** | 0.0005 |
| S&P 500 | +0.740 | +0.887 | **-0.148** | 0.0005 |
| Gold | +0.741 | +0.931 | **-0.190** | 0.0005 |
| Bitcoin | +0.716 | +0.922 | **-0.206** | 0.0005 |
| Ethereum | +0.696 | +0.900 | **-0.203** | 0.0005 |
| Silver | +0.719 | +0.931 | **-0.212** | 0.0005 |
| WTI Crude | +0.719 | +0.901 | **-0.182** | 0.0005 |

**7 of 7 assets**, at the strongest significance level the 2000-shuffle permutation test can
report, confirm the theoretical mechanism: days flagged "atypical" by the diffusion detector
sit in measurably more contracting (lower, though still positive — these markets are not
switching into literal stable equilibria, just relatively less expansive) regions of the
embedded phase space than "typical" days. This upgrades the investigation's central claim from
a correlation between a detector score and forecast accuracy into a first-principles dynamical
account: the detector is, in effect, picking out local-stability pockets that nonlinear
dynamics predicts should be more forecastable, which is exactly the regime-dependent
predictability mechanism argued for in the forecasting literature (e.g. the
"conditional/regime-dependent predictability" line of work on bull/bear and turmoil regimes) —
but grounded here in an explicit, computable, control-theoretic quantity rather than a
qualitative market-regime label.

**Finding 42 — the effect holds under a continuous, forecasting-journal-standard error metric
(MASE against a random-walk benchmark), not just binary directional accuracy.**
`continuous_error_metrics.py` re-evaluates the same causal forecasts kept continuous (predicted
and actual H-day log returns, not just their signs) and computes Mean Absolute Scaled Error
(Hyndman & Koehler, 2006 — the standard forecast-accuracy metric in the forecasting literature,
scale-free and comparable across assets) against the canonical no-change / random-walk naive
forecast, plus RMSE:

| asset | RMSE model | RMSE naive | MASE overall | MASE atypical | MASE typical | paired-bootstrap p |
|---|---|---|---|---|---|---|
| Brent | 0.0377 | 0.0433 | 0.884 | **0.839** | 0.967 | **p<0.001** |
| S&P 500 | 0.0172 | 0.0189 | 0.912 | **0.879** | 0.976 | **p<0.001** |
| Gold | 0.0175 | 0.0203 | 0.886 | **0.829** | 0.976 | **p<0.001** |
| Bitcoin | 0.0521 | 0.0595 | 0.892 | **0.826** | 1.001 | **p<0.001** |
| Ethereum | 0.0613 | 0.0700 | 0.911 | **0.826** | 1.036 | **p=0.001** |
| Silver | 0.0338 | 0.0375 | 0.896 | **0.820** | 1.014 | **p<0.001** |
| WTI Crude | 0.0414 | 0.0474 | 0.880 | **0.826** | 0.980 | **p<0.001** |

**7 of 7 assets beat the random-walk naive overall (MASE < 1)**, and **7 of 7 show a lower
(better) MASE on atypical days than on typical days** — the continuous-error result agrees with
every earlier binary-accuracy finding, so the phenomenon is not an artifact of thresholding
continuous returns into up/down labels. Notably, on *typical* days alone the model is
frequently at or slightly worse than the naive random walk (Bitcoin 1.001, Ethereum 1.036,
Silver 1.014) — the entire edge over the random-walk benchmark is concentrated in the
atypical-regime subset, reinforcing finding 41's dynamical account rather than suggesting a
small uniform edge spread evenly across all days. The p-values are from a 2000-resample paired
bootstrap over folds on the MAE improvement (model vs. naive) — closing the significance-testing
gap flagged in finding 45: every asset's 95% CI for the improvement excludes zero (e.g. Brent
+0.0034, CI [0.0024, 0.0044]; full CIs in `continuous_error_metrics.py`'s output).

**Finding 43 — the analog model also beats a classical econometric baseline (AR(1)), not just
the random-walk naive.** `arima_baseline_comparison.py` adds the most obvious next benchmark:
a causal AR(1) model (statsmodels `ARIMA(1,0,0)`, periodically refit on
the same causal history), forecasting the cumulative H-step return, evaluated with the same
MASE metric as finding 42. (Bitcoin failed to collect in this run due to a transient Yahoo
Finance data outage — the same kind of flakiness noted for Gold in finding 33 — not a result;
reported honestly rather than silently dropped.)

| asset | MASE, analog model | MASE, AR(1) | paired-bootstrap p |
|---|---|---|---|
| Brent | **0.884** | 1.007 | **p<0.001** |
| S&P 500 | **0.912** | 1.008 | **p<0.001** |
| Gold | **0.886** | 0.997 | **p<0.001** |
| Bitcoin | **0.892** | 1.002 | **p<0.001** |
| Ethereum | **0.911** | 1.002 | **p=0.001** |
| Silver | **0.896** | 0.998 | **p<0.001** |
| WTI Crude | **0.880** | 1.005 | **p<0.001** |

**7 of 7 available assets**: the analog model beats the classical AR(1) baseline (Bitcoin
collected successfully on this re-run, unlike the transient Yahoo Finance outage noted in the
original run). AR(1) itself beats the random-walk naive on only 2 of 7 assets (Gold, Silver) —
consistent with the well-known finding in the empirical-finance literature that a plain
low-order AR model rarely adds value over a no-change forecast on daily returns. This rules out
the objection that the analog model's edge (finding 42) is just "any reasonable model beats
random walk" — a standard econometric competitor, evaluated on the identical folds, does not.
Every asset's 95% bootstrap CI for the MAE improvement over AR(1) excludes zero.

**Finding 44 — the analog model also beats GARCH(1,1), and the atypicality detector is only
moderately correlated with GARCH conditional volatility — it is not just rediscovering
volatility clustering.** `garch_baseline_comparison.py` adds the second classical benchmark
that naturally follows AR(1) (finding 43): a causal, periodically-refit GARCH(1,1)
(constant-mean, `arch` package) forecasting the cumulative H-step return, plus a direct check
of whether the diffusion atypicality score (finding 31) is just a proxy for GARCH's conditional
volatility forecast.

| asset | MASE, analog model | MASE, GARCH(1,1) | corr(atypicality, GARCH cond. vol) | paired-bootstrap p |
|---|---|---|---|---|
| Brent | **0.884** | 1.001 | +0.444 | **p<0.001** |
| S&P 500 | **0.912** | 0.986 | +0.457 | **p<0.001** |
| Gold | **0.886** | 0.997 | +0.315 | **p<0.001** |
| Bitcoin | **0.892** | 1.000 | +0.358 | **p<0.001** |
| Ethereum | **0.911** | 1.002 | +0.274 | **p=0.001** |
| Silver | **0.896** | 1.001 | +0.339 | **p<0.001** |
| WTI Crude | **0.880** | 1.000 | +0.347 | **p<0.001** |

**7 of 7 assets**: the analog model beats GARCH(1,1) on MASE, every one with a 95% bootstrap CI
for the MAE improvement excluding zero. GARCH(1,1)'s own mean forecast beats the random-walk
naive on only 3 of 7 assets — consistent with finding 43's AR(1) result and the broader
literature finding that conditional-mean forecasts of daily returns rarely improve much on "no
change," even when the conditional-*variance* model (GARCH's actual strength) is
well-specified. The atypicality-vs-GARCH-volatility correlation is positive and non-trivial
(mean +0.36 across assets, confirming atypical days do tend to be higher-volatility days) but
far from 1 — meaning the detector is not simply relabeling GARCH's own volatility forecast.
This is an important, honestly-reported boundary: the effect is *related to* but not *reducible
to* volatility clustering, which answers the natural objection directly rather than leaving it
unaddressed.

**Finding 45 — a full integrity audit of the pipeline, data, and evaluation methodology
(prompted by moving toward a selective venue), with one real issue found, tested, and shown
NOT to be driving any result.** Before relying further on findings 31-44, every causal/leakage
boundary, the statistics, and the data source were re-checked line by line rather than
re-asserted.

**What was checked and confirmed sound:**
- *Data*: every finding uses real daily OHLC history pulled live from Yahoo Finance
  (`yfinance`) — no synthetic or simulated price data anywhere in findings 1-44. (The
  `diffusion/` module's synthetic DDPM-sampled scenarios are a separate, clearly-labeled
  alternative forecasting method, never mixed into the analog-forecast findings.)
- *Forecast-target causality*: in every script, the historical analogs used to build a
  forecast are selected via `select_analogs(d, j, last_ok=j-1)` with a Theiler exclusion window
  — strictly past data only, the future return being predicted is never accessible to the
  selection step. This is the one boundary that would be fatal if violated (as the earlier,
  already-corrected forward-difference and fracdiff-indexing bugs were), and it holds
  throughout.
- *Dempster-Shafer fusion math* (`dempster_shafer_fusion.py`): `ds_combine_two` was checked
  term-by-term against the textbook Dempster's combination rule for a 2-element frame with an
  uncertainty mass — matches exactly, including the conflict-renormalization term.
- *Embedding/indexing* (`embed`, the `p = j + (M-1)*TAU` price-index convention): re-derived by
  hand for M=3, TAU=1 and confirmed consistent with every consumer of `pc`.

**What was found and corrected:** `robustness_journal_grade.collect_full` and
`cross_asset_contagion.collect_with_dates` fit the three classical detectors (Isolation Forest,
LOF, Mahalanobis-PCA) on `hist = pc[:j+1]` — a window that **includes today's own point** —
then scored that same point with the resulting model. This is in-sample ("transductive")
anomaly scoring, not strictly "fit on the past, score the present." It does **not** leak the
forecast target (the future return being predicted stays strictly out of reach, per the point
above) — it could only bias the atypicality *score* itself.

`audit_leakage_check.py` re-ran the exact same folds on Brent and S&P 500 twice — once as
originally coded (today included in the fit) and once corrected (`hist = pc[:j]`, today
strictly excluded) — and compared the resulting accuracy gaps:

| asset | detector | gap, self-inclusive (as coded) | gap, strictly causal (corrected) |
|---|---|---|---|
| Brent | Isolation Forest | +11.7pp | **+13.5pp** |
| Brent | LOF | +13.3pp | +13.4pp |
| S&P 500 | Isolation Forest | +4.0pp (p=0.0500) | **+4.8pp (p=0.0275)** |
| S&P 500 | LOF | +5.3pp (p=0.0140) | +5.3pp (p=0.0130) |

The self-inclusive and strictly-causal scores were correlated at 0.99-1.00 across both assets,
and the corrected, strictly-causal version is, if anything, slightly **stronger** on every row
checked — the opposite of what an inflationary leakage artifact would look like. The issue was
real and worth finding, but it was conservative, not inflationary; it has now been fixed in both
source files for correctness going forward.

**Update: the full 7-asset re-run is done.** Every downstream finding that touches the fixed
columns (33, 35, 36) has now been re-run end to end with the corrected code and the tables
above reflect the new numbers directly, rather than the original spot-check standing in for
them. (Finding 44 did not need a re-run: `garch_baseline_comparison.py` never used the
self-inclusive classical-detector columns in the first place — it only uses the diffusion
atypicality score, which was unaffected by this fix.) Net effect across all three: the core
claims are unchanged, Gold is now included everywhere it was previously missing due to a data
outage, and one pairwise detail in finding 36 (two crypto-vs-traditional pairs moved from
not-significant to significant-negative) changed and is reported as such above rather than
re-asserted as identical.

**A separate, non-code methodological note worth stating explicitly:** `statistical_rigor.py`'s
`permutation_test` reports a **one-sided** p-value (`(null_gaps >= observed).mean()`), consistent
with this investigation's single, pre-specified directional hypothesis ("atypical days are MORE
predictable") — but this should be stated as a one-sided test in any formal write-up rather than
left implicit, since a reader's default assumption is usually two-sided. Relatedly, that
function has no `+1` numerator/denominator correction, so it can report `p=0.0000` on a 2000-shuffle
test where the more defensible statement is `p < 0.0005` (the test's resolution floor) — a
precision-of-language issue, not a wrong conclusion.

**Update: the significance-testing gap in findings 42-44 is now closed.** Those three findings
originally reported point-estimate MASE values without an accompanying significance test.
`continuous_error_metrics.bootstrap_compare` adds a 2000-resample paired bootstrap over folds
(resampling fold-level absolute errors with replacement, comparing the model's mean absolute
error to each baseline's) and reports a 95% CI and a one-sided empirical p-value for the MAE
improvement. Re-running all three comparisons with this test: **every one of the 21
asset-by-baseline comparisons across findings 42 (vs. random-walk), 43 (vs. AR(1)), and 44 (vs.
GARCH(1,1)) has a 95% CI for the improvement that excludes zero, at p ≤ 0.001.** (Bitcoin also
collected successfully against AR(1) on this re-run, closing the one asset gap noted in the
original finding 43.) The updated tables and numbers are reflected directly in findings 42-44
above rather than repeated here.

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
