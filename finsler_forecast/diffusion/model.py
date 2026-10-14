from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ContextEncoder(nn.Module):
    """Pools a context window of any length down to a fixed-size condition vector."""

    def __init__(self, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, cond_dim),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx.unsqueeze(1))


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm = nn.GroupNorm(8, channels)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.conv(x))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None]) + beta[:, :, None]
        return x + self.act(h)


class DenoiseNet(nn.Module):
    """FiLM-conditioned 1D-CNN predicting the diffusion noise for a horizon of log-returns."""

    def __init__(
        self,
        horizon_len: int,
        time_dim: int = 64,
        cond_dim: int = 64,
        channels: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.horizon_len = horizon_len
        self.time_dim = time_dim
        self.ctx_encoder = ContextEncoder(cond_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.cond_merge = nn.Linear(cond_dim * 2, cond_dim)
        self.in_proj = nn.Conv1d(1, channels, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList([FiLMBlock(channels, cond_dim) for _ in range(n_blocks)])
        self.out_proj = nn.Conv1d(channels, 1, kernel_size=5, padding=2)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        c_emb = self.ctx_encoder(ctx)
        cond = self.cond_merge(torch.cat([t_emb, c_emb], dim=-1))
        h = self.in_proj(x_noisy.unsqueeze(1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h).squeeze(1)
