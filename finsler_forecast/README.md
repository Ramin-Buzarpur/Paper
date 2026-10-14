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

## Layout

```
config.py      asset universe + trading calendar per category
data.py        yfinance price download
formatting.py  price display helpers
geometry.py    delay embedding, local Finsler metric, analog-flow engine
plotting.py    Plotly dashboard figure
ui.py          Streamlit page
app.py         entry point
```

## Run

```bash
pip install -r requirements.txt
streamlit run finsler_forecast/app.py
```
