from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from diffusion.forecast import DiffusionForecast
from formatting import price_decimals
from geometry import FinslerRandersGeodesicEngine

BIAS_LABELS = {
    "bullish": "bullish analog bias",
    "bearish": "bearish analog bias",
    "neutral": "no significant analog bias",
}

ANALOG_SCENARIO_COLOR = "rgba(0, 240, 255, 0.25)"
ANALOG_CENTER_STYLES = [
    ("frechet_mean_price", "Fréchet mean (Finsler, wind-aware)", "#39FF14", None),
    ("karcher_mean_price", "Karcher mean (Riemannian)", "#00F0FF", "dash"),
    ("manifold_median_price", "Geometric median (Finsler L1)", "#FFE600", "dot"),
]

DIFFUSION_SCENARIO_COLOR = "rgba(255, 102, 255, 0.15)"
DIFFUSION_MAX_PLOTTED_SCENARIOS = 30


def _subsample_rows(arr: np.ndarray, max_rows: int) -> np.ndarray:
    if arr.shape[0] <= max_rows:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_rows).astype(int)
    return arr[idx]


def build_forecast_figure(
    engine: FinslerRandersGeodesicEngine,
    diffusion: DiffusionForecast | None = None,
    show_analog: bool = True,
) -> go.Figure:
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

    if show_analog:
        for k in range(engine.k_used):
            fig.add_trace(
                go.Scatter(
                    x=engine.forecast_dates,
                    y=engine.scenario_prices[k],
                    mode="lines",
                    line=dict(color=ANALOG_SCENARIO_COLOR, width=1.2),
                    showlegend=(k == 0),
                    name=f"analog scenarios (K={engine.k_used})",
                )
            )
        for attr, name, color, dash in ANALOG_CENTER_STYLES:
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

    if diffusion is not None:
        plotted = _subsample_rows(diffusion.scenario_prices, DIFFUSION_MAX_PLOTTED_SCENARIOS)
        for k in range(plotted.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=diffusion.forecast_dates,
                    y=plotted[k],
                    mode="lines",
                    line=dict(color=DIFFUSION_SCENARIO_COLOR, width=1.0),
                    showlegend=(k == 0),
                    name=f"diffusion scenarios (N={diffusion.scenario_prices.shape[0]})",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=diffusion.forecast_dates,
                y=diffusion.mean_price_path,
                mode="lines+markers",
                line=dict(color="#FF3DFF", width=3.0),
                marker=dict(size=5, color="#FF3DFF"),
                name="diffusion mean",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=diffusion.forecast_dates,
                y=diffusion.median_price_path,
                mode="lines+markers",
                line=dict(color="#FF3DFF", width=2.0, dash="dot"),
                marker=dict(size=5, color="#FF3DFF"),
                name="diffusion median",
            )
        )

    methods = [m for m, used in [("Finsler analog flow", show_analog), ("diffusion", diffusion is not None)] if used]
    title = f"{engine.asset_name} — {' vs '.join(methods)} forecast"
    if show_analog:
        title += f"<br>wind strength ||b||_a = {engine.wind_magnitude:.3f} | {BIAS_LABELS[engine.wind_direction_bias]}"

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis=dict(title="date"),
        yaxis=dict(title="price", tickformat=f".{price_decimals(float(engine.prices[-1]))}f"),
        height=650,
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig
