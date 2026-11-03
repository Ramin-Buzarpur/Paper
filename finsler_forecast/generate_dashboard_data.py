"""
Recomputes the headline numbers behind research_dashboard.py and writes them to
dashboard_data.json, so the Streamlit dashboard can load instantly instead of re-running every
backtest live. Re-run this whenever an underlying experiment changes.

    python finsler_forecast/generate_dashboard_data.py
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

from data import download_close_prices
from geometry import FinslerRandersGeodesicEngine

HORIZON = 3
N_FOLDS = 80
STEP_BACK = 2
MIN_HISTORY = 200
N_BLOCKS = 8
K_TEST = 2

ASSETS = [
    ("BZ=F", "Brent Crude Oil", "B"),
    ("^GSPC", "S&P 500", "B"),
    ("BTC-USD", "Bitcoin", "D"),
]


def download_with_volume(ticker, years):
    df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=False)
    df = df[["Close", "Volume"]].dropna()
    df = df[df["Close"] > 0].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def collect_folds(ticker, name, freq, n_scenarios=7, n_boot=200):
    rng = np.random.default_rng(0)  # fresh per asset, matches the originally-reported bootstrap-gate figures
    df = download_with_volume(ticker, 6)
    prices = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    vol_mean = pd.Series(volume).rolling(20, min_periods=5).mean().to_numpy()
    vol_std = pd.Series(volume).rolling(20, min_periods=5).std().to_numpy()
    vol_z = (volume - vol_mean) / np.where(vol_std > 0, vol_std, 1.0)

    n_total = len(prices)
    latest = n_total - HORIZON - 1
    origins = sorted(sorted(range(latest, MIN_HISTORY, -STEP_BACK)[:N_FOLDS]))

    rows = []
    for origin in origins:
        current_price, actual = prices[origin], prices[origin + HORIZON]
        try:
            engine = (
                FinslerRandersGeodesicEngine(ticker=ticker, asset_name=name, freq=freq,
                                              drop_last_bar=False, n_scenarios=n_scenarios)
                .fetch_and_reconstruct(df[["Close"]].iloc[: origin + 1])
                .simulate(HORIZON)
            )
        except Exception:
            continue
        if engine.k_used != n_scenarios:
            continue
        analog_log_ret = np.log(engine.scenario_prices[:, -1] / current_price)
        boot_spread = float(rng.choice(analog_log_ret, size=(200, n_scenarios), replace=True).mean(axis=1).std())
        rows.append({
            "current_price": current_price, "actual_future_price": actual,
            "actual_up": actual > current_price,
            "frechet_pred": float(engine.frechet_mean_price[-1]),
            "karcher_pred": float(engine.karcher_mean_price[-1]),
            "naive_pred": current_price,
            "vol_z": vol_z[origin], "boot_spread": boot_spread,
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def dir_acc(df, pred_col):
    up = df[pred_col] > df["current_price"]
    return float((up == df["actual_up"]).mean())


def cpcv_dist(df, pred_col):
    n = len(df)
    edges = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    blocks = [np.arange(edges[b], edges[b + 1]) for b in range(N_BLOCKS)]
    accs = []
    for combo in itertools.combinations(range(N_BLOCKS), K_TEST):
        idx = np.concatenate([blocks[b] for b in combo])
        sub = df.iloc[idx]
        if len(sub) < 5:
            continue
        accs.append(dir_acc(sub, pred_col))
    return accs


def causal_bucket(df, col, min_train=20):
    vals = df[col].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        bucket[i] = "high" if vals[i] >= np.median(vals[:i]) else "low"
    out = df.copy()
    out["bucket"] = bucket
    return {
        b: {"dir_acc": dir_acc(out[out.bucket == b], "karcher_pred"), "n": int((out.bucket == b).sum())}
        for b in ("high", "low")
    }


def main():
    result = {}
    for ticker, name, freq in ASSETS:
        print(f"=== {name} ===")
        df = collect_folds(ticker, name, freq)
        entry = {
            "ticker": ticker,
            "n_folds": len(df),
            "frechet_dir_acc": dir_acc(df, "frechet_pred"),
            "karcher_dir_acc": dir_acc(df, "karcher_pred"),
            "naive_dir_acc": dir_acc(df, "naive_pred"),
            "karcher_cpcv": cpcv_dist(df, "karcher_pred"),
            "naive_cpcv": cpcv_dist(df, "naive_pred"),
            "volume_gate": causal_bucket(df, "vol_z"),
            "bootstrap_gate": causal_bucket(df, "boot_spread"),  # note: "high" boot_spread = less confident
        }
        result[name] = entry
        print(json.dumps({k: v for k, v in entry.items() if k not in ("karcher_cpcv", "naive_cpcv")}, indent=2))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_data.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
