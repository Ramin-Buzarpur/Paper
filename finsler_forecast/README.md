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

## Run

```bash
pip install -r requirements.txt
streamlit run finsler_forecast/app.py
```
