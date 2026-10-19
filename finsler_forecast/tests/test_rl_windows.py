from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsler_forecast.rl.agent import PPOAgent, WindowEnv, buy_hold_fold, gae, run_fold, train_ppo
from finsler_forecast.rl.backtest import summarize, window_metrics
from finsler_forecast.rl.dataset import MIN_WINDOW_BARS, window_origins


def _window(origin: pd.Timestamp, n: int = 20, ret: float = 0.001, prior: float = 0.004) -> pd.DataFrame:
    rets = np.full(n, ret, dtype=np.float32)
    return pd.DataFrame(
        {
            "origin": origin,
            "timestamp": pd.date_range(origin, periods=n, freq="h"),
            "close": 1.1 * np.exp(np.cumsum(rets)),
            "ret": rets,
            "is_closed": np.zeros(n, dtype=bool),
            "feat_vol": np.full(n, 0.001, dtype=np.float32),
            "feat_momentum": np.full(n, ret * 24, dtype=np.float32),
            "frechet_ret": np.full(n, prior),
            "karcher_ret": np.full(n, prior / 2),
            "median_ret": np.full(n, prior),
        }
    )


def make_windows(n_windows: int = 4, n_bars: int = 20) -> pd.DataFrame:
    origins = pd.date_range("2024-01-01", periods=n_windows, freq="3D")
    return pd.concat([_window(o, n_bars) for o in origins], ignore_index=True)


def test_env_state_dim_and_step():
    env = WindowEnv(make_windows(), prior_scale=0.01)
    s = env.reset()
    assert s.shape == (env.state_dim,)
    assert env.state_dim == 6
    s2, r, done, info = env.step(np.zeros(1, dtype=np.float32))
    assert isinstance(r, float) and np.isfinite(r)
    assert "port_ret" in info and s2.shape == s.shape


def test_prior_signal_is_capped_and_position_vol_scaled():
    env = WindowEnv(make_windows(), prior_scale=0.01)
    env.reset()
    assert env.data.prior_for(env.origin, env.prior_col, env.prior_scale) == pytest.approx(0.4)  # 0.004 / 0.01
    assert env.target_position(0.0) == pytest.approx(0.4)  # signal * size (size == 1 at target vol)
    assert env.target_position(10.0) == pytest.approx(1.0)  # residual capped by tanh + signal clip


def test_missing_prior_falls_back_to_zero():
    windows = make_windows()
    windows.loc[windows["origin"] == windows["origin"].iloc[0], "frechet_ret"] = np.nan
    env = WindowEnv(windows, prior_scale=0.01)
    assert env.data.prior_for(env.origins[0], env.prior_col, env.prior_scale) == 0.0


def test_window_ends_after_n_bars_and_charges_entry_cost():
    env = WindowEnv(make_windows(n_bars=20), prior_scale=0.01, cost_bps=5.0)
    origin = env.origins[0]
    fold = run_fold(env, None, origin)
    assert len(fold["port_rets"]) == 19
    assert fold["turnovers"][0] == pytest.approx(0.4)  # first trade pays 5 bps * 0.4 notional


def test_constant_positive_window_earns_its_prior_return():
    env = WindowEnv(make_windows(n_bars=20), prior_scale=0.01, cost_bps=0.0)
    fold = run_fold(env, None, env.origins[0])
    assert fold["port_rets"].sum() == pytest.approx(0.4 * 0.001 * 19, rel=1e-6)


def test_buy_hold_pays_cost_once():
    rets = np.full(10, 0.001, dtype=np.float32)
    out = buy_hold_fold(rets, cost_bps=5.0)
    assert out[0] == pytest.approx(0.001 - 0.0005)
    assert out[1:].tolist() == rets[1:].tolist()


def test_gae_shapes_and_finiteness():
    adv = gae([0.1, -0.2, 0.05], [0.0, 0.1, 0.2], [False, False, True], last_value=0.0, gamma=0.99, lam=0.95)
    assert len(adv) == 3 and np.isfinite(adv).all()


def test_ppo_update_runs_with_finite_stats():
    env = WindowEnv(make_windows(n_windows=6), prior_scale=0.01)
    agent = PPOAgent(env.state_dim, hidden=16, layers=1, lr=1e-3)
    history = train_ppo(env, agent, total_updates=2, steps_per_update=32)
    assert all(np.isfinite(v) for v in history[-1]["stats"].values())


def test_window_origins_are_spaced_by_calendar_days():
    idx = pd.date_range("2024-01-01", periods=24 * 9, freq="h")
    hourly = pd.Series(1.1, index=idx)
    origins = window_origins(hourly, window_days=3)
    assert len(origins) == 3  # 9 days -> starts at day 0, 3, 6
    assert [(pd.Timestamp(b) - pd.Timestamp(a)).days for a, b in zip(origins, origins[1:])] == [3, 3]


def test_short_windows_are_not_dropped_from_a_built_frame():
    # MIN_WINDOW_BARS is applied by build_windows, which needs the real parquet files;
    # the env itself must still accept a window of any length.
    env = WindowEnv(make_windows(n_bars=MIN_WINDOW_BARS + 1), prior_scale=0.01)
    assert len(run_fold(env, None, env.origins[0])["port_rets"]) == MIN_WINDOW_BARS


def test_summarize_orders_by_sharpe_and_keeps_window_count():
    rows = []
    for origin, ret in zip(pd.date_range("2024-01-01", periods=3, freq="3D"), (0.001, -0.001, 0.002)):
        rows.append({"origin": origin, "method": "good", "returns": np.full(10, ret, dtype=np.float32)})
        rows.append({"origin": origin, "method": "flat", "returns": np.zeros(10, dtype=np.float32)})
    table = summarize(pd.DataFrame(rows))
    assert table.iloc[0]["method"] == "good"
    assert table["n_windows"].tolist() == [3, 3]


def test_window_metrics_drawdown_is_negative_only():
    m = window_metrics(np.array([0.01, -0.02, 0.005], dtype=np.float32))
    assert 0.0 <= m["max_drawdown"] < 1.0
    assert m["total_return"] < 0.0
