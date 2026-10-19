"""
Trains a lightweight conditional DDPM on hourly EUR/USD log-returns and saves
a checkpoint that `forecast.py` can load to sample future scenarios.

Design:
  - context_len hourly bars -> condition vector (ContextEncoder)
  - horizon_len hourly bars -> the sequence the model learns to denoise
  - horizon_len is derived from the data (median bars/day * horizon_multiplier
    days), not hardcoded, since forex hourly data has weekend gaps
  - standard DDPM: fixed linear beta schedule, model predicts the noise
    (Ho et al. 2020), FiLM conditioning at every conv block
  - context_len, horizon_len and the normalization stats are saved inside the
    checkpoint so callers never need to duplicate this logic

Trains on the denoised dataset by default (see diffusion/denoise.py) — pass
dataset_variant="raw" to train on the unfiltered data instead, e.g. to compare
the two in the UI.

Usage:
    python -m finsler_forecast.diffusion.train
    python -m finsler_forecast.diffusion.train --variant raw
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .dataset import DATA_DIR, DATASET_VARIANTS, load_hourly_log_returns
from .model import DenoiseNet


def checkpoint_path(variant: str = "denoised") -> str:
    if variant not in DATASET_VARIANTS:
        raise ValueError(f"unknown dataset variant {variant!r}; choices: {sorted(DATASET_VARIANTS)}")
    return os.path.join(DATA_DIR, f"diffusion_eurusd_{variant}.pt")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows(log_ret: np.ndarray, context_len: int, horizon_len: int) -> tuple[np.ndarray, np.ndarray]:
    total = context_len + horizon_len
    n_windows = len(log_ret) - total + 1
    if n_windows <= 0:
        raise ValueError("not enough data for the requested context/horizon length")
    ctx = np.zeros((n_windows, context_len), dtype=np.float32)
    hor = np.zeros((n_windows, horizon_len), dtype=np.float32)
    for i in range(n_windows):
        ctx[i] = log_ret[i : i + context_len]
        hor[i] = log_ret[i + context_len : i + total]
    return ctx, hor


@dataclass
class DiffusionConfig:
    context_len: int
    horizon_len: int
    bars_per_day: float
    n_timesteps: int = 200
    mean: float = 0.0
    std: float = 1.0


class GaussianDiffusion:
    def __init__(self, n_timesteps: int, device: torch.device):
        self.T = n_timesteps
        betas = torch.linspace(1e-4, 0.02, n_timesteps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self.alphas_cumprod[t].sqrt()[:, None]
        sqrt_1m_ac = (1 - self.alphas_cumprod[t]).sqrt()[:, None]
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, ctx: torch.Tensor, horizon_len: int) -> torch.Tensor:
        device = ctx.device
        n = ctx.shape[0]
        x = torch.randn(n, horizon_len, device=device)
        for t_idx in reversed(range(self.T)):
            t = torch.full((n,), t_idx, device=device, dtype=torch.long)
            eps_pred = model(x, t, ctx)
            alpha_t, alpha_bar_t, beta_t = self.alphas[t_idx], self.alphas_cumprod[t_idx], self.betas[t_idx]
            mean = (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred) / torch.sqrt(alpha_t)
            x = mean + torch.sqrt(beta_t) * torch.randn_like(x) if t_idx > 0 else mean
        return x


def train(
    dataset_variant: str = "denoised",
    context_len: int = 168,
    horizon_multiplier: int = 3,
    n_timesteps: int = 200,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 2e-4,
    out_path: str | None = None,
    log_ret: np.ndarray | None = None,
    bars_per_day: float | None = None,
) -> None:
    device = get_device()
    if log_ret is None:
        log_ret, bars_per_day = load_hourly_log_returns(DATASET_VARIANTS[dataset_variant])
    if out_path is None:
        out_path = checkpoint_path(dataset_variant)
    horizon_len = int(round(bars_per_day * horizon_multiplier))
    print(f"bars/day ~= {bars_per_day:.1f} -> horizon_len = {horizon_len}, context_len = {context_len}")

    mean, std = float(log_ret.mean()), float(log_ret.std() + 1e-12)
    log_ret_norm = (log_ret - mean) / std
    ctx_all, hor_all = make_windows(log_ret_norm, context_len, horizon_len)
    print(f"training windows: {len(ctx_all)}")

    ctx_t = torch.tensor(ctx_all, device=device)
    hor_t = torch.tensor(hor_all, device=device)
    model = DenoiseNet(horizon_len=horizon_len).to(device)
    diffusion = GaussianDiffusion(n_timesteps, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(ctx_t)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x0, ctx = hor_t[idx], ctx_t[idx]
            t = torch.randint(0, n_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_noisy = diffusion.q_sample(x0, t, noise)
            loss = nn.functional.mse_loss(model(x_noisy, t, ctx), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch:3d}/{epochs}  loss={total_loss / n_batches:.5f}")

    config = DiffusionConfig(context_len, horizon_len, bars_per_day, n_timesteps, mean, std)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": asdict(config)}, out_path)
    print(f"saved checkpoint: {out_path}")


def load_model_and_config(path: str | None = None) -> tuple[DenoiseNet, DiffusionConfig]:
    device = get_device()
    ckpt = torch.load(path or checkpoint_path(), map_location=device)
    config = DiffusionConfig(**ckpt["config"])
    model = DenoiseNet(horizon_len=config.horizon_len).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="denoised", choices=sorted(DATASET_VARIANTS))
    args = parser.parse_args()
    train(dataset_variant=args.variant)
