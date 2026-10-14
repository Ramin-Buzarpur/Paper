# PEDC Alpha — Diffusion + RL Trading Engine

Hybrid forecasting and portfolio engine: a conditional diffusion model (MaTCHS) predicts
short-horizon return scenarios across a basket of stocks, a PPO reinforcement-learning
agent learns a residual position-sizing policy on top of that, and a walk-forward
backtester turns the combined signal into a core-satellite portfolio.

> Research prototype. Backtest results are historical and do not guarantee future
> performance. Not financial advice.

## Architecture

```
market data (OHLCV, 4 exchanges) ──┐
news sentiment (optional) ─────────┤
                                    ▼
                    feature engineering (momentum, MACD, ATR, vol z-score)
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
      diffusion model        RL policy (PPO)       classical momentum
      (MaTCHS denoiser,    residual position     21-day z-score
       masked relational    sizing on top of
       attention, DDIM      momentum, vol
       sampling)            targeting, drawdown
              │              penalty                       │
              └──────── weighted signal combination ────────┘
                                    ▼
                    core-satellite portfolio construction
                    (core: trend + vol targeting,
                     satellite: combined alpha, 20% of risk budget)
                                    ▼
                         walk-forward backtest with costs
```

## Layout

```
configs/default.yaml        all hyperparameters
src/
  config.py                 YAML config loader
  pipeline.py                data + feature preparation, device setup
  data/
    market_data.py           scans raw OHLCV files, picks the most liquid symbols, caches panels
    features.py               momentum/MACD/vol features
    graph.py                  pairwise correlation graph for the relational attention mask
    news.py                    optional news sentiment scoring
    dataset.py                 diffusion training dataset
  models/
    temporal.py                temporal encoder, sinusoidal time embedding
    matchs.py                   MaTCHS: masked relational transformer
    diffusion.py                 conditional DDPM + DDIM sampling
    rl_agent.py                   portfolio env + PPO agent + GAE
  training/
    train_diffusion.py            diffusion training with early stopping
    train_rl.py                    RL training, model selection on validation Sharpe
  backtest/
    engine.py                      cost-aware backtest, core-satellite logic
    metrics.py                     Sharpe / Sortino / Calmar / max drawdown
    walkforward.py                  no-lookahead out-of-sample evaluation
  inference/
    predictor.py                   generates forward return scenarios
    strategy.py                     combines diffusion + RL + momentum into a signal
scripts/                     prepare_cache, update_data, run_training, run_backtest, live_signal, plot_forecast
tests/                       unit tests for each module above
```

`data/`, `data_cache/` and `outputs/` are not checked in — they are generated locally
(see `.gitignore`) and should never be committed.

## Setup

Requires Python 3.13.

```bash
pip install -r requirements.txt
```

Place raw OHLCV CSVs under `data/raw/<market>/csv/*.csv` (or change `data.raw_dir` in
`configs/default.yaml`), then:

```bash
python3 scripts/prepare_cache.py     # scan markets, pick liquid symbols, cache panels
python3 scripts/run_training.py      # train diffusion model, then RL agent
python3 scripts/run_backtest.py      # walk-forward backtest + report + plots
python3 -m pytest tests/ -q          # run the test suite
```

Daily signal generation, once a model is trained:

```bash
python3 scripts/update_data.py
python3 scripts/live_signal.py --capital 10000 --top 10
```

## Known limitations

- Net daily alpha after transaction costs is close to zero on large-cap US equities;
  most of the value is in the drawdown-controlled core, not the satellite.
- News coverage is limited to a short recent window, so sentiment is effectively zero
  in historical backtests and only contributes live.
- Periodic retraining is needed as the universe and market regime drift
  (`run_training.py` every few months, `run_training.py --only rl` weekly for a lighter
  policy refresh).
