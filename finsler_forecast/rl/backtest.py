"""
Walk-forward RL backtest over the 3-day windows of rl/dataset.py.

The agent is trained once on the earliest windows and then evaluated
deterministically, window by window, on every later window: at each step it has
only seen the past, exactly as it would live. The Finsler forecast (and, when
available, the diffusion forecast) is the base signal the policy learns a
residual on top of, so the geometry is never replaced — it is compared against
the policy that improves on it.

Usage:
    python -m finsler_forecast.rl.backtest                     # denoised dataset
    python -m finsler_forecast.rl.backtest raw                 # raw dataset
    python -m finsler_forecast.rl.backtest denoised --diffusion
    python -m finsler_forecast.rl.backtest --compare-variants
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .agent import PPOAgent, WindowEnv, WindowSet, buy_hold_fold, run_fold, train_ppo
from .dataset import WINDOW_DAYS, build_windows

HORIZON = WINDOW_DAYS
ANNUALIZATION = 252.0 / HORIZON  # windows per year
PRIOR_SCALE = 0.01  # a 1% forecasted window return maps to a full-size position
PPO_CONFIG = {"hidden": 32, "layers": 1, "lr": 3e-4, "epochs": 4, "entropy_coef": 0.01}
PPO_UPDATES = 6
PPO_STEPS_PER_UPDATE = 256
METHOD_PRIORS = {
    "finsler_frechet": "frechet_ret",
    "finsler_karcher": "karcher_ret",
    "finsler_median": "median_ret",
    "diffusion_mean": "diffusion_mean_ret",
    "diffusion_median": "diffusion_median_ret",
}
RL_METHOD = "rl_finsler_frechet"


def _env(data: WindowSet, origins: list[pd.Timestamp], prior_col: str, random_start: bool = False) -> WindowEnv:
    return WindowEnv(data, origins=origins, prior_col=prior_col, prior_scale=PRIOR_SCALE, random_start=random_start)


def train_agent(data: WindowSet, train_origins: list[pd.Timestamp], prior_col: str, seed: int) -> PPOAgent:
    torch.manual_seed(seed)
    env = _env(data, train_origins, prior_col, random_start=True)
    agent = PPOAgent(env.state_dim, **PPO_CONFIG)
    torch.nn.init.zeros_(agent.net.mu_head.weight)
    torch.nn.init.zeros_(agent.net.mu_head.bias)  # start as the prior, learn the residual from there
    train_ppo(env, agent, PPO_UPDATES, PPO_STEPS_PER_UPDATE)
    return agent

def window_metrics(port_rets: np.ndarray) -> dict:
    """Metrics for one contiguous return path (all test windows, in time order)."""
    equity = np.exp(np.log1p(port_rets).cumsum())
    sd = float(np.std(port_rets))
    return {
        "total_return": float(equity[-1] - 1.0),
        "sharpe": float(np.mean(port_rets) / sd * np.sqrt(ANNUALIZATION)) if sd > 1e-12 else 0.0,
        "hit_rate": float((port_rets > 0).mean()),
        "max_drawdown": float(np.max(1.0 - equity / np.maximum.accumulate(equity))),
    }


def run_backtest(
    variant: str = "denoised",
    with_diffusion: bool = False,
    n_train_windows: int = 60,
    n_test_windows: int | None = None,
    seed: int = 42,
    windows: pd.DataFrame | None = None,
    retrain_each_window: bool = False,
) -> pd.DataFrame:
    """One row per (test window, method) holding that window's bar-level portfolio returns.

    `retrain_each_window` refits PPO before every test window using only the windows
    before it; it is the stricter (and much slower) variant of the same walk-forward.
    """
    if windows is None:
        windows = build_windows(variant, with_diffusion=with_diffusion)
    data = windows if isinstance(windows, WindowSet) else WindowSet(windows)
    origins = [o for o in data.origins if pd.notna(data.prior[o]["frechet_ret"].iloc[0])]
    if len(origins) <= n_train_windows:
        raise ValueError(f"need more than {n_train_windows} windows with a prior, got {len(origins)}")

    methods = [m for m, col in METHOD_PRIORS.items() if col in windows.columns and windows[col].notna().any()]
    test_origins = origins[n_train_windows:] if n_test_windows is None else origins[n_train_windows:][-n_test_windows:]
    agent = None if retrain_each_window else train_agent(data, origins[:n_train_windows], "frechet_ret", seed)

    rows = []
    for i, origin in enumerate(test_origins, 1):
        window_rets = windows.loc[windows["origin"] == origin, "ret"].to_numpy()
        for method in methods:
            eval_env = _env(data, [origin], METHOD_PRIORS[method])
            rows.append({"origin": origin, "method": method, "returns": run_fold(eval_env, None, origin)["port_rets"]})
            if method == "finsler_frechet":
                if retrain_each_window:
                    agent = train_agent(data, origins[: origins.index(origin)], "frechet_ret", seed)
                rows.append({"origin": origin, "method": RL_METHOD, "returns": run_fold(eval_env, agent, origin)["port_rets"]})
                rows.append({"origin": origin, "method": "buy_hold", "returns": buy_hold_fold(window_rets)})
        if i % 25 == 0 or i == len(test_origins):
            print(f"  window {i}/{len(test_origins)} ({pd.Timestamp(origin).date()})")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-window returns into one row per method."""
    out = []
    for method, group in results.groupby("method"):
        ordered = group.sort_values("origin")  # keep the windows in time order
        rets = np.concatenate(ordered["returns"].to_numpy())
        row = window_metrics(rets)
        row |= {
            "method": method,
            "n_windows": len(group),
            "mean_window_return": float(ordered["returns"].apply(np.sum).mean()),
        }
        out.append(row)
    return pd.DataFrame(out).sort_values("sharpe", ascending=False).reset_index(drop=True)


def compare_variants(n_train_windows: int = 60, n_test_windows: int | None = None,
                     with_diffusion: bool = False, seed: int = 42) -> pd.DataFrame:
    """Run the same walk-forward test on the raw and denoised datasets and stack the summaries."""
    summaries = []
    for variant in ("denoised", "raw"):
        print(f"[{variant}] building windows...")
        # diffusion sampling dominates the build cost, so only the windows that will be tested get it
        last_n = None if n_test_windows is None else n_train_windows + n_test_windows
        windows = build_windows(variant, with_diffusion=with_diffusion, last_n_origins=last_n)
        print(f"[{variant}] {windows['origin'].nunique()} windows, running walk-forward...")
        summary = summarize(run_backtest(variant, with_diffusion, n_train_windows, n_test_windows, seed, windows))
        summary.insert(0, "dataset", variant)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", default="denoised", choices=["raw", "denoised"])
    parser.add_argument("--diffusion", action="store_true", help="also score the diffusion prior")
    parser.add_argument("--train-windows", type=int, default=60)
    parser.add_argument("--test-windows", type=int, default=None)
    parser.add_argument("--retrain", action="store_true", help="refit PPO before every test window")
    parser.add_argument("--compare-variants", action="store_true", help="run both datasets and compare")
    args = parser.parse_args()

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    if args.compare_variants:
        table = compare_variants(args.train_windows, args.test_windows, args.diffusion)
    else:
        last_n = None if args.test_windows is None else args.train_windows + args.test_windows
        windows = build_windows(args.variant, with_diffusion=args.diffusion, last_n_origins=last_n)
        results = run_backtest(
            args.variant, args.diffusion, args.train_windows, args.test_windows, windows=windows,
            retrain_each_window=args.retrain,
        )
        table = summarize(results)
    print()
    print(table.to_string(index=False))
