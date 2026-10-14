"""
Walk-forward backtest: does picking a different center of the scenario ensemble
actually forecast better than naive persistence, 3 trading days ahead? Compares
MAE and directional accuracy. If a diffusion checkpoint is available for the
ticker (see diffusion/train.py), its mean/median forecast is scored too.

Usage:
    python finsler_forecast/backtest.py "EUR/USD"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import MARKET_CATEGORIES
from data import download_close_prices
from diffusion.dataset import hourly_log_return_series
from diffusion.forecast import DiffusionScenarioForecaster, diffusion_available
from geometry import FinslerRandersGeodesicEngine


def run_backtest(
    ticker: str,
    asset_name: str,
    years: int = 5,
    n_scenarios: int = 7,
    horizon_days: int = 3,
    n_folds: int = 20,
    step_back: int = 5,
    diffusion: DiffusionScenarioForecaster | None = None,
    hourly_returns: pd.Series | None = None,
) -> pd.DataFrame:
    price_df = download_close_prices(ticker, years)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    dates = price_df.index
    n_total = len(prices)

    min_history = 200  # above the engine's own 150-bar minimum, with margin
    latest_origin = n_total - horizon_days - 1
    if latest_origin <= min_history:
        raise ValueError("not enough history to backtest with this configuration")

    origins = sorted(range(latest_origin, min_history, -step_back)[:n_folds])

    rows = []
    for origin in origins:
        current_price = prices[origin]
        actual_future_price = prices[origin + horizon_days]
        actual_up = actual_future_price > current_price

        try:
            engine = (
                FinslerRandersGeodesicEngine(ticker=ticker, asset_name=asset_name, years=years, n_scenarios=n_scenarios)
                .fetch_and_reconstruct(price_df.iloc[: origin + 1])
                .simulate(horizon_days)
            )
        except Exception as exc:
            print(f"  [skip] origin {dates[origin].date()}: {exc}")
            continue

        row = {
            "origin_date": dates[origin],
            "current_price": current_price,
            "actual_future_price": actual_future_price,
            "actual_up": actual_up,
            "frechet_pred": float(engine.frechet_mean_price[-1]),
            "karcher_pred": float(engine.karcher_mean_price[-1]),
            "median_pred": float(engine.manifold_median_price[-1]),
            "naive_pred": current_price,
        }

        if diffusion is not None and hourly_returns is not None:
            try:
                ctx = hourly_returns.loc[: dates[origin]].to_numpy(dtype=np.float64)
                fc = diffusion.forecast(ctx, current_price, engine.forecast_dates, horizon_days)
                row["diffusion_mean_pred"] = fc.mean_price_path[-1]
                row["diffusion_median_pred"] = fc.median_price_path[-1]
            except Exception as exc:
                print(f"  [diffusion skip] origin {dates[origin].date()}: {exc}")

        rows.append(row)

    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> pd.DataFrame:
    methods = [c[: -len("_pred")] for c in df.columns if c.endswith("_pred")]
    rows = []
    for m in methods:
        col = f"{m}_pred"
        err = df[col] - df["actual_future_price"]
        pred_up = df[col] > df["current_price"]
        rows.append(
            {
                "method": m,
                "MAE": err.abs().mean(),
                "RMSE": float(np.sqrt((err**2).mean())),
                "directional_accuracy": float((pred_up == df["actual_up"]).mean()),
                "n_folds": int(df[col].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE")


if __name__ == "__main__":
    import sys

    asset_name = sys.argv[1] if len(sys.argv) > 1 else "EUR/USD"
    ticker = MARKET_CATEGORIES["Forex Majors"].get(asset_name, "EURUSD=X")

    diffusion_forecaster, hourly_returns = None, None
    if diffusion_available(ticker):
        diffusion_forecaster = DiffusionScenarioForecaster()
        hourly_returns = hourly_log_return_series()
        print("diffusion checkpoint found, including it in the comparison")
    else:
        print("no diffusion checkpoint for this ticker, comparing analog methods only")

    print(f"running walk-forward backtest for {asset_name} ({ticker})...")
    df = run_backtest(ticker=ticker, asset_name=asset_name, diffusion=diffusion_forecaster, hourly_returns=hourly_returns)

    if df.empty:
        print("no valid folds produced")
        sys.exit(1)

    print(f"\ncompleted {len(df)} folds from {df['origin_date'].min().date()} to {df['origin_date'].max().date()}\n")
    result = score(df)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    print(result.to_string(index=False))

    naive_mae = result.loc[result["method"] == "naive", "MAE"].values[0]
    best = result.iloc[0]
    print(f"\nbest method by MAE: {best['method']} (MAE={best['MAE']:.5f})")
    if best["method"] != "naive":
        improvement = (naive_mae - best["MAE"]) / naive_mae * 100
        print(f"=> {best['method']} beat naive persistence by {improvement:.1f}% MAE")
