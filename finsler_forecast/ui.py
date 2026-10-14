from __future__ import annotations

import pandas as pd
import streamlit as st

from config import CATEGORY_CALENDAR, MARKET_CATEGORIES
from data import download_close_prices
from formatting import fmt_price
from geometry import FinslerRandersGeodesicEngine
from plotting import BIAS_LABELS, build_forecast_figure

FORECAST_HORIZON_DAYS = 3


@st.cache_data(ttl=3600, show_spinner=False)
def cached_close_prices(ticker: str, years: int) -> pd.DataFrame:
    return download_close_prices(ticker, years)


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


def render_results(engine: FinslerRandersGeodesicEngine) -> None:
    curr = float(engine.prices[-1])
    fr = float(engine.frechet_mean_price[-1])
    ka = float(engine.karcher_mean_price[-1])
    md = float(engine.manifold_median_price[-1])
    pct = lambda v: (v - curr) / curr * 100.0

    st.subheader(f"Summary — {engine.asset_name}")
    show_dataframe(
        pd.DataFrame(
            [
                {
                    "Asset": engine.asset_name,
                    "Ticker": engine.ticker,
                    "Last close": fmt_price(curr),
                    "Analog bias": BIAS_LABELS[engine.wind_direction_bias],
                    "Wind strength": f"{engine.wind_magnitude:.3f}",
                    "T+3 Fréchet": f"{fmt_price(fr)} ({pct(fr):+.2f}%)",
                    "T+3 Karcher": f"{fmt_price(ka)} ({pct(ka):+.2f}%)",
                    "T+3 median": f"{fmt_price(md)} ({pct(md):+.2f}%)",
                }
            ]
        )
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wind strength", f"{engine.wind_magnitude:.3f}", delta=BIAS_LABELS[engine.wind_direction_bias], delta_color="off")
    c2.metric("Fréchet mean (T+3)", fmt_price(fr), delta=f"{pct(fr):+.2f}%")
    c3.metric("Karcher mean (T+3)", fmt_price(ka), delta=f"{pct(ka):+.2f}%")
    c4.metric("Geometric median (T+3)", fmt_price(md), delta=f"{pct(md):+.2f}%")
    st.caption(
        f"embedding: m={engine.m}, τ={engine.tau}, {len(engine.point_cloud)} states | "
        f"k={engine.k_neighbors} neighbours | K={engine.k_used} analog scenarios"
    )

    tab1, tab2 = st.tabs(["Analog flow & centers", "Financial wind"])
    with tab1:
        show_plot(build_forecast_figure(engine))
        st.caption(
            "scenario paths are real historical continuations of similar states, not integrated "
            "geodesics; in this flat chart the Karcher mean is the arithmetic mean of the scenario "
            "states, and the Fréchet mean departs from it only through the wind."
        )
    with tab2:
        st.markdown(
            r"Randers function $F(x,y)=\sqrt{a_{ij}y^iy^j}+b_iy^i$, $b=-w$ (moving with the wind $w$ "
            r"is cheaper), $\|b\|_a=\tanh(\rho)\le 0.95$ where $\rho$ is the neighbours' mean forward "
            r"drift in units of their local flow dispersion."
        )
        w = engine.latest_wind_covector
        show_dataframe(
            pd.DataFrame(
                {
                    "coordinate": ["r(t)", "r(t-τ)", "r(t-2τ)"],
                    "wind covector": [f"{x:+.4f}" for x in w],
                }
            )
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

    years = st.sidebar.slider("History (years)", 1, 20, 5)
    n_scenarios = st.sidebar.slider("Analog scenarios (K)", 3, 20, 7)
    k_neighbors = st.sidebar.slider("Local neighbourhood (k)", 10, 100, 30, 5)
    tau = st.sidebar.slider("Delay τ (0 = auto)", 0, 10, 1)
    drop_last = st.sidebar.checkbox("Drop today's unfinished bar", value=True)

    if st.sidebar.button("Run", type="primary"):
        with st.spinner(f"computing analog flow for {asset_name}..."):
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
                st.session_state["engine"] = engine
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
        render_results(engine)
