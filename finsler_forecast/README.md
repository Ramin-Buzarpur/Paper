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

## Layout

```
config.py          asset universe + trading calendar per category
data.py             yfinance price download
formatting.py        price display helpers
geometry.py           delay embedding, local Finsler metric, analog-flow engine
plotting.py            Plotly dashboard figure (analog and/or diffusion)
ui.py                   Streamlit page
app.py                   entry point
backtest.py              walk-forward comparison across all methods
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
