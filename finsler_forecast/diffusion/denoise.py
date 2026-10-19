"""
Detects and removes closed-market "flat" candles from minute OHLCV data: bars
with zero volume and High == Low, which happen during weekends, holidays, and
other periods when no trades occurred. Dukascopy still emits a row for these
periods, so they show up as fake zero-volatility bars unless filtered out —
left in, they get resampled into hourly bars that teach the diffusion model a
nonexistent "nothing happens on weekends" pattern.

Usage:
    python -m finsler_forecast.diffusion.denoise EURUSD_minute_dukascopy.csv
"""

from __future__ import annotations

import pandas as pd

MIN_CLOSURE_MINUTES = 30


def load_minute_csv(path: str) -> pd.DataFrame:
    dtypes = {"Open": "float32", "High": "float32", "Low": "float32", "Close": "float32", "Volume": "float32"}
    df = pd.read_csv(path, dtype=dtypes, parse_dates=["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)


def flat_candle_mask(df: pd.DataFrame) -> pd.Series:
    return (df["Volume"] == 0) & (df["High"] == df["Low"])


def detect_closures(df: pd.DataFrame, min_duration_minutes: int = MIN_CLOSURE_MINUTES) -> pd.DataFrame:
    """Groups consecutive flat candles into closure periods and keeps the ones at least `min_duration_minutes` long."""
    is_flat = flat_candle_mask(df)
    flat_df = df[is_flat]
    # cumsum over non-flat rows gives consecutive flat rows the same group id (run-length encoding)
    run_id = (~is_flat).cumsum()[is_flat]

    closures = flat_df.groupby(run_id).agg(
        start_time=("Datetime", "first"),
        end_time=("Datetime", "last"),
        flat_minutes=("Datetime", "count"),
        price=("Close", "first"),
    )
    return closures[closures["flat_minutes"] >= min_duration_minutes].reset_index(drop=True)


def remove_flat_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Drops every zero-volume flat row, regardless of how long the closure lasted."""
    return df[~flat_candle_mask(df)].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "EURUSD_minute_dukascopy.csv"
    df = load_minute_csv(csv_path)
    closures = detect_closures(df)
    closures.to_csv("market_closures_report.csv", index=False)
    print(f"{len(df)} bars loaded, {len(closures)} closures >= {MIN_CLOSURE_MINUTES} min, saved to market_closures_report.csv")
