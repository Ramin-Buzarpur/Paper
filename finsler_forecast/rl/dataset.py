"""
Hourly bars split into fixed-length trading windows (3 days by default), for the
walk-forward RL backtest. Each window carries its Finsler prior, i.e. the forecast
the geometry engine makes from the daily closes strictly before the window opens.

The raw and denoised minute datasets share every trading-hour close; they differ
only in the closed-market hours the raw variant still contains. Building the
windows on hourly bars is therefore what makes the raw-vs-denoised comparison
measurable.

Usage:
    python -m finsler_forecast.rl.dataset            # build both variants, 3-day windows
    python -m finsler_forecast.rl.dataset raw        # just the raw variant
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..diffusion.dataset import DATASET_VARIANTS
from ..diffusion.forecast import DIFFUSION_CHECKPOINTS, DiffusionScenarioForecaster, available_variants
from ..geometry import FinslerRandersGeodesicEngine

WINDOW_DAYS = 3
HISTORY_DAYS = 250  # daily closes handed to the Finsler engine for its prior
FEATURE_LOOKBACK = 24
MIN_WINDOW_BARS = 6  # holiday-shortened windows below this are dropped


def hourly_closes(parquet_path: str) -> pd.Series:
    """Hourly last-close series (one bar per clock hour that actually traded)."""
    minute_df = pd.read_parquet(parquet_path, columns=["Datetime", "Close"])
    return minute_df.set_index("Datetime")["Close"].resample("1h").last().dropna()


def daily_closes(hourly: pd.Series) -> pd.Series:
    """Business-day closes taken from the hourly series (same values in both variants)."""
    daily = hourly.resample("1D").last().dropna()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    return daily[daily.index.dayofweek < 5]


def window_origins(hourly: pd.Series, window_days: int = WINDOW_DAYS) -> list[pd.Timestamp]:
    """Start of each non-overlapping calendar window, on a fixed date grid from the first bar.

    Windows are `window_days` calendar days apart, so both dataset variants see the
    same periods; a window's length is however many hourly bars actually traded in
    it (fewer across weekends and holidays).
    """
    if len(hourly) == 0:
        return []
    grid = pd.date_range(hourly.index[0].ceil("D"), hourly.index[-1], freq=f"{window_days}D")
    starts = hourly.index.searchsorted(grid)
    return [hourly.index[i] for i in starts if i < len(hourly)]


def compute_finsler_priors(daily: pd.Series, origins: list[pd.Timestamp], horizon_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Finsler forecast at each window start, using only daily closes before that start.

    A window whose prior can't be computed (too little history) is absent from the
    frame; `build_windows` then leaves its prior columns as NaN.
    """
    rows = []
    for origin in origins:
        history = daily[daily.index < origin]
        if len(history) < HISTORY_DAYS:
            continue
        window = history.iloc[-HISTORY_DAYS:]
        try:
            engine = (
                FinslerRandersGeodesicEngine(ticker="", asset_name="", freq="B", drop_last_bar=False)
                .fetch_and_reconstruct(window.to_frame())
                .simulate(horizon_days)
            )
        except ValueError:
            continue
        last_price = float(engine.prices[-1])
        rows.append(
            {
                "origin": origin,
                "frechet_ret": float(np.log(engine.frechet_mean_price[-1] / last_price)),
                "karcher_ret": float(np.log(engine.karcher_mean_price[-1] / last_price)),
                "median_ret": float(np.log(engine.manifold_median_price[-1] / last_price)),
                "wind_strength": float(engine.wind_magnitude),
                "next_ret_t": float(engine.latest_geometry.next_ret_t),
            }
        )
    return pd.DataFrame(rows)


def diffusion_priors(
    origins: list[pd.Timestamp], hourly: pd.Series, forecaster: DiffusionScenarioForecaster, n_scenarios: int = 100
) -> pd.DataFrame:
    """Diffusion mean/median window return at each origin, conditioned on the hourly history before it."""
    config = forecaster.config
    days = max(1, int(round(config.horizon_len / config.bars_per_day)))
    log_returns = np.log(hourly).diff().dropna()
    rows = []
    for origin in origins:
        history = log_returns[log_returns.index < origin]
        if len(history) < config.context_len:
            continue
        current_price = float(hourly[hourly.index < origin].iloc[-1])
        dates = pd.DatetimeIndex([origin + pd.Timedelta(days=i) for i in range(days + 1)])
        try:
            fc = forecaster.forecast(history.iloc[-config.context_len :].to_numpy(), current_price, dates, days, n_scenarios)
        except ValueError:
            continue
        rows.append(
            {
                "origin": origin,
                "diffusion_mean_ret": float(np.log(fc.mean_price_path[-1] / current_price)),
                "diffusion_median_ret": float(np.log(fc.median_price_path[-1] / current_price)),
            }
        )
    return pd.DataFrame(rows)


def build_windows(
    variant: str = "denoised",
    window_days: int = WINDOW_DAYS,
    with_diffusion: bool = False,
    last_n_origins: int | None = None,
) -> pd.DataFrame:
    """One row per hourly bar in each window, with the prior attached at the window start.

    Columns: origin, timestamp, close, ret, is_closed, the three feature columns, and
    the prior columns (NaN when unavailable for that origin). `last_n_origins` keeps
    only the most recent windows — useful with `with_diffusion`, whose sampling is the
    slow part.
    """
    if variant not in DATASET_VARIANTS:
        raise ValueError(f"unknown dataset variant {variant!r}; choices: {sorted(DATASET_VARIANTS)}")

    hourly = hourly_closes(DATASET_VARIANTS[variant])
    daily = daily_closes(hourly)
    origins = window_origins(hourly, window_days)
    if not origins:
        raise ValueError("no complete windows in this dataset")
    if last_n_origins is not None:
        origins = origins[-last_n_origins:]

    priors = compute_finsler_priors(daily, origins, window_days)
    if with_diffusion and variant in available_variants("EURUSD=X"):
        forecaster = DiffusionScenarioForecaster(DIFFUSION_CHECKPOINTS["EURUSD=X"][variant])
        priors = priors.merge(diffusion_priors(origins, hourly, forecaster), on="origin", how="outer")

    log_ret = np.log(hourly).diff().fillna(0.0)
    vol = log_ret.rolling(FEATURE_LOOKBACK).std().fillna(0.0)
    momentum = log_ret.rolling(FEATURE_LOOKBACK).sum().fillna(0.0)

    starts = np.searchsorted(hourly.index.to_numpy(), np.array(origins, dtype="datetime64[ns]"))
    bounds = np.append(starts, len(hourly))
    frames = []
    for i, origin in enumerate(origins):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi - lo < MIN_WINDOW_BARS:
            continue
        window = hourly.iloc[lo:hi]
        frames.append(
            pd.DataFrame(
                {
                    "origin": origin,
                    "timestamp": window.index,
                    "close": window.to_numpy(dtype=np.float64),
                    "ret": log_ret.iloc[lo:hi].to_numpy(dtype=np.float64),
                    "is_closed": log_ret.iloc[lo:hi].to_numpy() == 0.0,
                    "feat_vol": vol.iloc[lo:hi].to_numpy(dtype=np.float64),
                    "feat_momentum": momentum.iloc[lo:hi].to_numpy(dtype=np.float64),
                }
            )
        )
    if not frames:
        raise ValueError("no window has enough bars; check the dataset")
    rows = pd.concat(frames, ignore_index=True)
    return rows.merge(priors, on="origin", how="left")


if __name__ == "__main__":
    import sys

    variant = sys.argv[1] if len(sys.argv) > 1 else "denoised"
    df = build_windows(variant, with_diffusion=True)
    print(f"{variant}: {df['origin'].nunique()} windows, {len(df)} hourly bars")
    print(f"prior available for {df.drop_duplicates('origin')['frechet_ret'].notna().sum()} windows")
