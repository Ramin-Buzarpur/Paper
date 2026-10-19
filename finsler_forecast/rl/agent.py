"""
Single-asset windowed PPO agent for the Finsler forecast stack.

The geometry engine's forecast is not replaced: it enters each window as the
base signal, and PPO learns a residual on top of it (same idea as the
momentum-residual policy in src/models/rl_agent.py). Episodes are the short
windows built by rl/dataset.py (3 calendar days of hourly bars), so the agent
sees history window by window instead of one long series.

State: [bar return, rolling vol, rolling momentum, prior signal, position, drawdown]
Action: residual in [-1, 1], added to the prior signal.
Position: signal in [-1, 1], scaled down when realized vol is above the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

BARS_PER_DAY = 24.0


class WindowSet:
    """The windows of one dataset variant, split once into per-origin arrays."""

    def __init__(self, windows: pd.DataFrame):
        self.origins = list(windows["origin"].drop_duplicates())
        self.rets, self.feats, self.vol, self.closed = {}, {}, {}, {}
        self.prior = {}
        for origin, window in windows.groupby("origin", sort=True):
            rets = window["ret"].to_numpy(dtype=np.float32)
            vol = np.maximum(window["feat_vol"].to_numpy(dtype=np.float32) * np.sqrt(BARS_PER_DAY), 1e-5)
            self.rets[origin] = rets
            self.vol[origin] = vol
            self.closed[origin] = window["is_closed"].to_numpy(dtype=bool)
            self.feats[origin] = np.column_stack(
                [rets, np.log(vol), window["feat_momentum"].to_numpy(dtype=np.float32)]
            ).astype(np.float32)
            self.prior[origin] = window

    def prior_for(self, origin: pd.Timestamp, column: str, scale: float) -> float:
        value = float(self.prior[origin][column].iloc[0])
        return 0.0 if not np.isfinite(value) else float(np.clip(value / scale, -1.0, 1.0))

    @property
    def state_dim(self) -> int:
        return int(next(iter(self.feats.values())).shape[1] + 3)


class WindowEnv:
    """Episodes = the fixed windows of one dataset variant, sampled in order or at random."""

    def __init__(
        self,
        windows: pd.DataFrame | WindowSet,
        origins: list[pd.Timestamp] | None = None,
        prior_col: str = "frechet_ret",
        prior_scale: float = 0.01,
        vol_target_annual: float = 0.15,
        cost_bps: float = 5.0,
        drawdown_penalty: float = 2.0,
        resid_scale: float = 1.0,
        random_start: bool = False,
        seed: int = 123,
    ):
        self.data = windows if isinstance(windows, WindowSet) else WindowSet(windows)
        if prior_col not in self.data.prior[self.data.origins[0]].columns:
            raise ValueError(f"missing prior column {prior_col!r}")
        self.origins = list(origins) if origins is not None else list(self.data.origins)
        self.cursor = -1
        self.prior_col = prior_col
        self.prior_scale = float(prior_scale)
        self.resid_scale = float(resid_scale)
        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(seed)
        self.vol_target_daily = float(vol_target_annual / np.sqrt(252.0))
        self.cost = float(cost_bps) / 1e4
        self.dd_pen = float(drawdown_penalty)
        self.reset()

    def _sample_origin(self) -> pd.Timestamp:
        if self.random_start:
            self.cursor = int(self.rng.integers(len(self.origins)))
        else:
            self.cursor = (self.cursor + 1) % len(self.origins)
        return self.origins[self.cursor]

    def reset(self) -> np.ndarray:
        self.origin = self._sample_origin()
        self.t = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        self.closed_bars = 0
        return self._state()

    @property
    def n_steps(self) -> int:
        return len(self.data.rets[self.origin]) - 1

    def _state(self) -> np.ndarray:
        f = self.data.feats[self.origin][self.t]
        prior = self.data.prior_for(self.origin, self.prior_col, self.prior_scale)
        extra = np.asarray([prior, self.position, self.drawdown], dtype=np.float32)
        return np.concatenate([f, extra]).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return self.data.state_dim

    @property
    def n_actions(self) -> int:
        return 1

    def target_position(self, action: float) -> float:
        prior = self.data.prior_for(self.origin, self.prior_col, self.prior_scale)
        signal = prior + self.resid_scale * float(np.tanh(action))
        size = min(1.0, self.vol_target_daily / float(self.data.vol[self.origin][self.t]))
        return float(np.clip(signal, -1.0, 1.0) * size)

    def step(self, action: np.ndarray):
        if self.t >= self.n_steps:
            return self._state(), 0.0, True, {"port_ret": 0.0, "equity": self.equity}

        new_pos = self.target_position(np.asarray(action, dtype=np.float32)[0])
        turnover = abs(new_pos - self.position)
        cost = turnover * self.cost
        port_ret = float(new_pos * self.data.rets[self.origin][self.t + 1] - cost)
        self.equity *= float(np.exp(np.log1p(np.clip(port_ret, -0.9, 10.0))))
        self.peak = max(self.peak, self.equity)
        new_dd = 1.0 - self.equity / self.peak
        reward = float(np.log1p(np.clip(port_ret, -0.9, 10.0)) - self.dd_pen * max(0.0, new_dd - self.drawdown))
        self.drawdown = new_dd
        self.position = new_pos
        self.closed_bars += int(self.data.closed[self.origin][self.t + 1])
        self.t += 1
        done = self.t >= self.n_steps
        info = {"port_ret": port_ret, "equity": self.equity, "turnover": turnover, "position": new_pos}
        return self._state(), reward, done, info


class _ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int, layers: int):
        super().__init__()
        mods: list[nn.Module] = []
        d_in = state_dim
        for _ in range(layers):
            mods += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Tanh()]
            d_in = hidden
        self.body = nn.Sequential(*mods)
        self.mu_head = nn.Linear(hidden, 1)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((1,), -1.0))

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        mu = torch.tanh(self.mu_head(h))
        v = self.v_head(h).squeeze(-1)
        return mu, v

    def dist(self, x: torch.Tensor) -> Normal:
        mu, _ = self.forward(x)
        std = torch.exp(self.log_std).clamp(0.05, 2.0)
        return Normal(mu, std)


def gae(rewards, values, dones, last_value, gamma: float, lam: float) -> np.ndarray:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for i in reversed(range(T)):
        next_v = last_value if i == T - 1 else values[i + 1]
        nonterm = 0.0 if dones[i] else 1.0
        delta = rewards[i] + gamma * next_v * nonterm - values[i]
        last = adv[i] = delta + gamma * lam * nonterm * last
    return adv


class PPOAgent:
    def __init__(self, state_dim: int, hidden: int = 32, layers: int = 1, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 epochs: int = 4, entropy_coef: float = 0.01, value_coef: float = 0.5,
                 device: torch.device | str = "cpu"):
        self.net = _ActorCritic(state_dim, hidden, layers).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = gae_lambda
        self.clip = clip_eps
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.net.dist(s)
        _, v = self.net(s)
        a = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return float(a.item()), float(v.item()), float(logp.item())

    def update(self, states, actions, advs, returns_, old_logp, batch_size: int = 256) -> dict:
        st = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        ac = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device).unsqueeze(-1)
        ad = torch.as_tensor(np.asarray(advs), dtype=torch.float32, device=self.device)
        rt = torch.as_tensor(np.asarray(returns_), dtype=torch.float32, device=self.device)
        ol = torch.as_tensor(np.asarray(old_logp), dtype=torch.float32, device=self.device)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        n = len(st)
        for _ in range(self.epochs):
            idx_all = np.arange(n)
            np.random.shuffle(idx_all)
            for b in range(0, n, batch_size):
                bi = idx_all[b : b + batch_size]
                dist = self.net.dist(st[bi])
                logp = dist.log_prob(ac[bi]).sum(-1)
                ratio = (logp - ol[bi]).exp()
                s1 = ratio * ad[bi]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ad[bi]
                pi_loss = -torch.min(s1, s2).mean()
                _, v = self.net(st[bi])
                v_loss = 0.5 * ((v - rt[bi]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                stats["pi_loss"] += float(pi_loss.item())
                stats["v_loss"] += float(v_loss.item())
                stats["entropy"] += float(ent.item())
        k = max(1, self.epochs * max(1, n // batch_size))
        return {key: v / k for key, v in stats.items()}


def collect_rollout(env: WindowEnv, agent: PPOAgent, steps: int):
    states, actions, logps, rewards, dones, values = [], [], [], [], [], []
    state = env.reset()
    while len(states) < steps:
        a, v, lp = agent.act(state)
        next_state, r, done, _ = env.step(np.asarray([a], dtype=np.float32))
        states.append(state)
        actions.append(a)
        logps.append(lp)
        rewards.append(r)
        dones.append(done)
        values.append(v)
        state = next_state
        if done:
            state = env.reset()
    last_v = values[-1] if not dones[-1] else 0.0
    adv = gae(rewards, values, dones, last_v, agent.gamma, agent.lam)
    returns_ = (adv + np.asarray(values)).astype(np.float32)
    return states, actions, adv, returns_, logps


def train_ppo(env: WindowEnv, agent: PPOAgent, total_updates: int, steps_per_update: int = 256):
    history = []
    for _ in range(total_updates):
        s, a, adv, r, lp = collect_rollout(env, agent, steps_per_update)
        stats = agent.update(s, a, adv, r, lp)
        history.append({"stats": stats, "n_steps": len(s), "equity": env.equity})
    return history


@torch.no_grad()
def run_fold(env: WindowEnv, agent: PPOAgent | None, origin: pd.Timestamp, residual: bool = True) -> dict:
    """Roll one window deterministically and return its bar-level portfolio returns.

    `agent=None` (or `residual=False`) trades the prior alone: the signal-only baseline.
    """
    env.origin, env.t = origin, 0
    env.position, env.equity, env.peak, env.drawdown, env.closed_bars = 0.0, 1.0, 1.0, 0.0, 0
    state = env._state()
    rets, turnovers, positions = [], [], []
    while True:
        if agent is None or not residual:
            a = 0.0
        else:
            a, _, _ = agent.act(state, deterministic=True)
        state, _, done, info = env.step(np.asarray([a], dtype=np.float32))
        rets.append(info["port_ret"])
        turnovers.append(info["turnover"])
        positions.append(info["position"])
        if done:
            break
    return {
        "origin": origin,
        "port_rets": np.asarray(rets, dtype=np.float32),
        "turnovers": np.asarray(turnovers, dtype=np.float32),
        "positions": np.asarray(positions, dtype=np.float32),
        "closed_share": float(env.closed_bars) / max(1, len(rets)),
    }


def buy_hold_fold(rets: np.ndarray, cost_bps: float = 5.0) -> np.ndarray:
    """Long-only buy-and-hold over the same bars: enter at the first bar paying cost, hold to the end."""
    out = rets.astype(np.float32).copy()
    out[0] = rets[0] - cost_bps / 1e4
    return out
