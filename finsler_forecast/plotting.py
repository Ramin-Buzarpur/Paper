from __future__ import annotations

import plotly.graph_objects as go

from formatting import price_decimals
from geometry import FinslerRandersGeodesicEngine

BIAS_LABELS = {
    "bullish": "bullish analog bias",
    "bearish": "bearish analog bias",
    "neutral": "no significant analog bias",
}

SCENARIO_COLOR = "rgba(0, 240, 255, 0.25)"
CENTER_STYLES = [
    ("frechet_mean_price", "Fréchet mean (Finsler, wind-aware)", "#39FF14", None),
    ("karcher_mean_price", "Karcher mean (Riemannian)", "#00F0FF", "dash"),
    ("manifold_median_price", "Geometric median (Finsler L1)", "#FFE600", "dot"),
]


def build_forecast_figure(engine: FinslerRandersGeodesicEngine) -> go.Figure:
    fig = go.Figure()
    hist_len = min(60, len(engine.prices))
    fig.add_trace(
        go.Scatter(
            x=engine.dates[-hist_len:],
            y=engine.prices[-hist_len:],
            mode="lines",
            line=dict(color="#A0AAB2", width=1.8),
            name="historical close",
        )
    )

    for k in range(engine.k_used):
        fig.add_trace(
            go.Scatter(
                x=engine.forecast_dates,
                y=engine.scenario_prices[k],
                mode="lines",
                line=dict(color=SCENARIO_COLOR, width=1.2),
                showlegend=(k == 0),
                name=f"analog scenarios (K={engine.k_used})",
            )
        )

    for attr, name, color, dash in CENTER_STYLES:
        line = dict(color=color, width=2.5 if dash else 3.0)
        if dash:
            line["dash"] = dash
        fig.add_trace(
            go.Scatter(
                x=engine.forecast_dates,
                y=getattr(engine, attr),
                mode="lines+markers",
                line=line,
                marker=dict(size=5, color=color),
                name=name,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=(
            f"{engine.asset_name} — Finsler analog flow forecast<br>"
            f"wind strength ||b||_a = {engine.wind_magnitude:.3f} | {BIAS_LABELS[engine.wind_direction_bias]}"
        ),
        xaxis=dict(title="date"),
        yaxis=dict(title="price", tickformat=f".{price_decimals(float(engine.prices[-1]))}f"),
        height=650,
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig
