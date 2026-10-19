from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import CATEGORY_CALENDAR, MARKET_CATEGORIES
from data import download_close_prices
from diffusion.dataset import DATASET_VARIANTS, hourly_log_return_series
from diffusion.forecast import (
    DIFFUSION_CHECKPOINTS,
    DiffusionForecast,
    DiffusionScenarioForecaster,
    available_variants,
)
from formatting import fmt_price
from geometry import FinslerRandersGeodesicEngine
from plotting import BIAS_LABELS, build_forecast_figure

FORECAST_HORIZON_DAYS = 3
ANALOG_METHOD = "Analog flow (Finsler)"
DIFFUSION_METHOD = "Diffusion scenarios"
COMPARE_METHOD = "Compare both"
VARIANT_LABELS = {"denoised": "Denoised (recommended)", "raw": "Raw (includes market-closure noise)"}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_close_prices(ticker: str, years: int) -> pd.DataFrame:
    return download_close_prices(ticker, years)


@st.cache_resource(show_spinner=False)
def cached_diffusion_forecaster(ticker: str, variant: str) -> DiffusionScenarioForecaster:
    return DiffusionScenarioForecaster(DIFFUSION_CHECKPOINTS[ticker][variant])


def show_dataframe(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:  # older Streamlit without the `width` kwarg
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_plot(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def render_results(
    engine: FinslerRandersGeodesicEngine,
    method: str,
    diffusion: DiffusionForecast | None,
    dataset_variant: str | None = None,
) -> None:
    show_analog = method in (ANALOG_METHOD, COMPARE_METHOD)
    curr = float(engine.prices[-1])
    pct = lambda v: (v - curr) / curr * 100.0

    st.subheader(f"Summary — {engine.asset_name}")
    summary = {"Asset": engine.asset_name, "Ticker": engine.ticker, "Last close": fmt_price(curr)}
    if show_analog:
        fr, ka, md = (
            float(engine.frechet_mean_price[-1]),
            float(engine.karcher_mean_price[-1]),
            float(engine.manifold_median_price[-1]),
        )
        summary |= {
            "Analog bias": BIAS_LABELS[engine.wind_direction_bias],
            "Wind strength": f"{engine.wind_magnitude:.3f}",
            "T+3 Fréchet": f"{fmt_price(fr)} ({pct(fr):+.2f}%)",
            "T+3 Karcher": f"{fmt_price(ka)} ({pct(ka):+.2f}%)",
            "T+3 median": f"{fmt_price(md)} ({pct(md):+.2f}%)",
        }
    if diffusion is not None:
        dm, dd = float(diffusion.mean_price_path[-1]), float(diffusion.median_price_path[-1])
        summary |= {
            "Dataset": VARIANT_LABELS.get(dataset_variant, dataset_variant),
            "T+3 diffusion mean": f"{fmt_price(dm)} ({pct(dm):+.2f}%)",
            "T+3 diffusion median": f"{fmt_price(dd)} ({pct(dd):+.2f}%)",
        }
    show_dataframe(pd.DataFrame([summary]))

    tabs = ["Forecast"]
    if show_analog:
        tabs.append("Financial wind")
    rendered = st.tabs(tabs)

    with rendered[0]:
        show_plot(build_forecast_figure(engine, diffusion=diffusion, show_analog=show_analog))
        captions = []
        if show_analog:
            captions.append(
                "analog scenario paths are real historical continuations of similar states, not "
                "integrated geodesics; the Karcher mean is the arithmetic mean of the scenario states "
                "in this flat chart, and the Fréchet mean departs from it only through the wind."
            )
        if diffusion is not None:
            captions.append(
                f"diffusion scenarios ({diffusion.scenario_prices.shape[0]} sampled) come from a DDPM "
                "trained on hourly log-returns, conditioned on the recent history."
            )
        if captions:
            st.caption(" ".join(captions))
        st.caption(
            f"embedding: m={engine.m}, τ={engine.tau}, {len(engine.point_cloud)} states | "
            f"k={engine.k_neighbors} neighbours | K={engine.k_used} analog scenarios"
        )

    if show_analog:
        with rendered[1]:
            st.markdown(
                r"Randers function $F(x,y)=\sqrt{a_{ij}y^iy^j}+b_iy^i$, $b=-w$ (moving with the wind "
                r"$w$ is cheaper), $\|b\|_a=\tanh(\rho)\le 0.95$ where $\rho$ is the neighbours' mean "
                r"forward drift in units of their local flow dispersion."
            )
            w = engine.latest_wind_covector
            show_dataframe(
                pd.DataFrame({"coordinate": ["r(t)", "r(t-τ)", "r(t-2τ)"], "wind covector": [f"{x:+.4f}" for x in w]})
            )
            g = engine.latest_geometry
            if g is not None:
                st.caption(
                    f"||b||_a = {g.strength:.3f} | analog next-day-return t-stat = {g.next_ret_t:+.2f} "
                    f"(|t|>2 labels bullish/bearish; neighbours overlap, so treat this as a heuristic)"
                )

    st.warning("research tool: not back-tested, no uncertainty estimate — not a trading signal on its own.")


def main() -> None:
    st.set_page_config(page_title="Finsler analog flow forecast", page_icon="🧭", layout="wide")
    st.title("Finsler-Randers analog flow forecast")

    st.sidebar.header("Parameters")
    market_cat = st.sidebar.selectbox("Asset class", list(MARKET_CATEGORIES.keys()))
    assets = MARKET_CATEGORIES[market_cat]
    asset_name = st.sidebar.selectbox("Asset", list(assets.keys()))
    ticker = assets[asset_name]

    variants = available_variants(ticker)
    method_options = [ANALOG_METHOD]
    if variants:
        method_options += [DIFFUSION_METHOD, COMPARE_METHOD]
    method = st.sidebar.selectbox("Forecast method", method_options)
    if not variants:
        st.sidebar.caption("diffusion scenarios need a trained checkpoint; none yet for this asset")

    years = st.sidebar.slider("History (years)", 1, 20, 5)
    n_scenarios = st.sidebar.slider("Analog scenarios (K)", 3, 20, 7)
    k_neighbors = st.sidebar.slider("Local neighbourhood (k)", 10, 100, 30, 5)
    tau = st.sidebar.slider("Delay τ (0 = auto)", 0, 10, 1)
    drop_last = st.sidebar.checkbox("Drop today's unfinished bar", value=True)
    n_diffusion_scenarios = 200
    dataset_variant = None
    if method in (DIFFUSION_METHOD, COMPARE_METHOD):
        n_diffusion_scenarios = st.sidebar.slider("Diffusion scenarios (N)", 20, 500, 200, 20)
        ordered = [v for v in ("denoised", "raw") if v in variants]
        dataset_variant = st.sidebar.selectbox(
            "Dataset", ordered, format_func=lambda v: VARIANT_LABELS.get(v, v)
        )

    if st.sidebar.button("Run", type="primary"):
        with st.spinner(f"computing {method.lower()} for {asset_name}..."):
            try:
                engine = (
                    FinslerRandersGeodesicEngine(
                        ticker=ticker,
                        asset_name=asset_name,
                        years=years,
                        k_neighbors=k_neighbors,
                        n_scenarios=n_scenarios,
                        tau=tau,
                        freq=CATEGORY_CALENDAR.get(market_cat, "B"),
                        drop_last_bar=drop_last,
                    )
                    .fetch_and_reconstruct(cached_close_prices(ticker, years))
                    .simulate(FORECAST_HORIZON_DAYS)
                )

                diffusion_forecast = None
                if method in (DIFFUSION_METHOD, COMPARE_METHOD):
                    forecaster = cached_diffusion_forecaster(ticker, dataset_variant)
                    hourly_returns = hourly_log_return_series(DATASET_VARIANTS[dataset_variant]).to_numpy(dtype=np.float64)
                    diffusion_forecast = forecaster.forecast(
                        hourly_returns,
                        current_price=float(engine.prices[-1]),
                        forecast_dates=engine.forecast_dates,
                        horizon_days=FORECAST_HORIZON_DAYS,
                        n_scenarios=n_diffusion_scenarios,
                    )

                st.session_state["engine"] = engine
                st.session_state["method"] = method
                st.session_state["diffusion_forecast"] = diffusion_forecast
                st.session_state["dataset_variant"] = dataset_variant
                st.session_state["error"] = None
            except Exception as exc:
                st.session_state["engine"] = None
                st.session_state["error"] = str(exc)

    engine = st.session_state.get("engine")
    error = st.session_state.get("error")
    if error:
        st.error(error)
    elif engine is None:
        st.info("choose an asset and click Run.")
    else:
        render_results(
            engine,
            st.session_state["method"],
            st.session_state["diffusion_forecast"],
            st.session_state.get("dataset_variant"),
        )
