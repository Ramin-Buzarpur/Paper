from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .dataset import DATASET_VARIANTS
from .train import DenoiseNet, DiffusionConfig, GaussianDiffusion, checkpoint_path, get_device, load_model_and_config

# ticker -> dataset variant -> checkpoint path; add a ticker once checkpoints are trained for it
DIFFUSION_CHECKPOINTS: dict[str, dict[str, str]] = {
    "EURUSD=X": {variant: checkpoint_path(variant) for variant in DATASET_VARIANTS},
}


def available_variants(ticker: str) -> list[str]:
    paths = DIFFUSION_CHECKPOINTS.get(ticker, {})
    return [variant for variant, path in paths.items() if os.path.exists(path)]


def diffusion_available(ticker: str, variant: str = "denoised") -> bool:
    return variant in available_variants(ticker)


@dataclass
class DiffusionForecast:
    forecast_dates: pd.DatetimeIndex    # length horizon_days + 1 (origin + each day)
    scenario_prices: np.ndarray         # (n_scenarios, horizon_days + 1)
    mean_price_path: np.ndarray         # (horizon_days + 1,)
    median_price_path: np.ndarray       # (horizon_days + 1,)


class DiffusionScenarioForecaster:
    """Wraps a trained diffusion checkpoint: recent hourly log-returns -> a day-by-day price forecast."""

    def __init__(self, path: str):
        self.model: DenoiseNet
        self.config: DiffusionConfig
        self.model, self.config = load_model_and_config(path)
        self.device = get_device()
        self.diffusion = GaussianDiffusion(self.config.n_timesteps, self.device)

    def forecast(
        self,
        recent_log_returns: np.ndarray,
        current_price: float,
        forecast_dates: pd.DatetimeIndex,
        horizon_days: int = 3,
        n_scenarios: int = 200,
    ) -> DiffusionForecast:
        c = self.config
        if len(recent_log_returns) < c.context_len:
            raise ValueError(f"need at least {c.context_len} recent hourly log returns for conditioning")
        if len(forecast_dates) != horizon_days + 1:
            raise ValueError("forecast_dates must have horizon_days + 1 entries")

        ctx_norm = (recent_log_returns[-c.context_len :] - c.mean) / c.std
        ctx_batch = torch.tensor(np.tile(ctx_norm, (n_scenarios, 1)), dtype=torch.float32, device=self.device)

        samples_norm = self.diffusion.sample(self.model, ctx_batch, c.horizon_len)
        hourly_returns = samples_norm.cpu().numpy() * c.std + c.mean  # (n_scenarios, horizon_len)

        # split the hourly horizon into `horizon_days` chunks -> one cumulative return per day
        daily_chunks = np.array_split(hourly_returns, horizon_days, axis=1)
        daily_returns = np.stack([chunk.sum(axis=1) for chunk in daily_chunks], axis=1)  # (n_scenarios, horizon_days)
        cum_returns = np.cumsum(daily_returns, axis=1)
        origin_col = np.zeros((n_scenarios, 1))
        prices = current_price * np.exp(np.concatenate([origin_col, cum_returns], axis=1))

        return DiffusionForecast(
            forecast_dates=forecast_dates,
            scenario_prices=prices,
            mean_price_path=prices.mean(axis=0),
            median_price_path=np.median(prices, axis=0),
        )
