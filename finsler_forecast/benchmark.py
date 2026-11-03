"""
Honest, nested walk-forward benchmark: karcher (the one Finsler-engine center that is ever
competitive, see README) vs. naive persistence vs. karcher + a small AI residual model
(ensemble.py), across several assets and two timeframes (daily, hourly).

Nesting / no-leakage rule: for fold i, the residual model and the blend weight are fit using
ONLY folds < i. Fold i itself is never used to pick its own weight or to train the model that
predicts it. This is why the first `min_train_folds` folds of every run are "warm-up" and
excluded from the final scoreboard.

Usage:
    python finsler_forecast/benchmark.py                      # all default assets, daily
    python finsler_forecast/benchmark.py --timeframe hourly
    python finsler_forecast/benchmark.py --asset BTC-USD --timeframe both
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf

from ensemble import FEATURE_NAMES, _direction_flip_model, _residual_model, engine_meta_features, pick_blend_weight
from geometry import FinslerRandersGeodesicEngine

warnings.filterwarnings("ignore")

DAILY_ASSETS = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "EURUSD=X": "EUR/USD",
    "BZ=F": "Brent Crude Oil", "^GSPC": "S&P 500",
}
HOURLY_ASSETS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "EURUSD=X": "EUR/USD"}


def download_prices(ticker: str, timeframe: str) -> pd.DataFrame:
    if timeframe == "daily":
        df = yf.Ticker(ticker).history(period="6y", interval="1d", auto_adjust=False)
    else:
        df = yf.Ticker(ticker).history(period="729d", interval="1h", auto_adjust=False)  # yfinance 1h cap
    df = df[["Close"]].dropna()
    df = df[df["Close"] > 0].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def collect_folds(
    price_df: pd.DataFrame, ticker: str, asset_name: str, horizon: int, n_folds: int, step_back: int, freq: str
) -> pd.DataFrame:
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    dates = price_df.index
    n_total = len(prices)
    min_history = 200
    latest_origin = n_total - horizon - 1
    if latest_origin <= min_history:
        return pd.DataFrame()

    origins = sorted(range(latest_origin, min_history, -step_back)[:n_folds])  # newest-first, then re-sort below
    origins = sorted(origins)  # chronological, required for nested (no-leakage) fitting later

    rows = []
    for origin in origins:
        current_price = prices[origin]
        actual_future_price = prices[origin + horizon]
        try:
            engine = (
                FinslerRandersGeodesicEngine(ticker=ticker, asset_name=asset_name, freq=freq, drop_last_bar=False)
                .fetch_and_reconstruct(price_df.iloc[: origin + 1])
                .simulate(horizon)
            )
        except Exception:
            continue

        karcher_pred = float(engine.karcher_mean_price[-1])
        actual_up = actual_future_price > current_price
        karcher_up = karcher_pred > current_price
        row = {
            "origin_date": dates[origin],
            "current_price": current_price,
            "actual_future_price": actual_future_price,
            "actual_up": actual_up,
            "naive_pred": current_price,
            "karcher_pred": karcher_pred,
            "karcher_err": karcher_pred - actual_future_price,
            "karcher_up": karcher_up,
            "karcher_wrong": int(karcher_up != actual_up),
        }
        row.update(engine_meta_features(engine))
        rows.append(row)

    return pd.DataFrame(rows)


def run_nested_ensemble(df: pd.DataFrame, min_train_folds: int = 15, val_frac: float = 0.3) -> pd.DataFrame:
    """Walk forward through df (already chronological); at fold i:
      1. split folds [0, i) into an earlier fit-set and a later, still-out-of-sample val-set,
      2. fit the residual model on the fit-set only, predict the val-set, and pick the blend
         weight from THAT (genuinely held-out) error — not from the data the model was fit on,
      3. refit the model on all of [0, i) and apply the chosen weight to score fold i.
    A weight picked on the same rows a model was trained on is leakage even when the test fold
    itself is held out; this is what step 2 avoids."""
    scored = []
    for i in range(min_train_folds, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i]

        n_val = max(5, int(len(train) * val_frac))
        fit_part, val_part = train.iloc[:-n_val], train.iloc[-n_val:]

        val_model = _residual_model()
        val_model.fit(fit_part[FEATURE_NAMES].to_numpy(), fit_part["karcher_err"].to_numpy())
        val_ai_pred_err = val_model.predict(val_part[FEATURE_NAMES].to_numpy())
        w = pick_blend_weight(val_part["karcher_err"].to_numpy(), val_ai_pred_err)

        model = _residual_model()
        model.fit(train[FEATURE_NAMES].to_numpy(), train["karcher_err"].to_numpy())
        test_ai_pred_err = float(model.predict(test[FEATURE_NAMES].to_numpy().reshape(1, -1))[0])
        # karcher_pred already *is* karcher's price prediction; the AI model predicts karcher's
        # *error* (pred - actual), so subtracting w * predicted_error corrects karcher toward actual.
        blended_pred = test["karcher_pred"] - w * test_ai_pred_err

        # Direction-flip gate: only ever touches the directional call, never the price level, so
        # it structurally cannot make MAE/RMSE worse than karcher's. Threshold (how confident the
        # classifier must be that karcher is wrong before flipping) is itself picked on val_part.
        clf = _direction_flip_model()
        clf.fit(fit_part[FEATURE_NAMES].to_numpy(), fit_part["karcher_wrong"].to_numpy())
        val_p_wrong = clf.predict_proba(val_part[FEATURE_NAMES].to_numpy())[:, 1]
        best_thresh, best_acc = 1.01, float((~val_part["karcher_wrong"].astype(bool)).mean())  # 1.01 = never flip
        for thresh in (0.55, 0.6, 0.65, 0.7, 0.75):
            flipped_wrong = np.where(val_p_wrong > thresh, 1 - val_part["karcher_wrong"].to_numpy(), val_part["karcher_wrong"].to_numpy())
            acc = float(1 - flipped_wrong.mean())
            if acc > best_acc + 0.02:  # same conservative margin logic as the price blend
                best_thresh, best_acc = thresh, acc

        clf_full = _direction_flip_model()
        clf_full.fit(train[FEATURE_NAMES].to_numpy(), train["karcher_wrong"].to_numpy())
        p_wrong = float(clf_full.predict_proba(test[FEATURE_NAMES].to_numpy().reshape(1, -1))[0, 1])
        dirflip_correct = (not bool(test["karcher_wrong"])) if p_wrong <= best_thresh else bool(test["karcher_wrong"])

        row = test.to_dict()
        row["blend_weight"] = w
        row["ai_blend_pred"] = blended_pred
        row["dirflip_correct"] = dirflip_correct
        scored.append(row)
    return pd.DataFrame(scored)


def confidence_gate_breakdown(df: pd.DataFrame, min_train_folds: int = 15) -> pd.DataFrame:
    """Selective-prediction test: does karcher do better on folds where the engine's own,
    causally-available confidence signal (|wind_t_stat|) was high, vs. low? For fold i the
    high/low split point is the median of |wind_t_stat| over folds [0, i) only (expanding,
    never peeking ahead) -- so "high confidence" is something you could have known *before*
    seeing the outcome, not a label chosen in hindsight."""
    conf = df["wind_t_stat"].abs().to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train_folds, len(df)):
        thresh = np.median(conf[:i])
        bucket[i] = "high_confidence" if conf[i] >= thresh else "low_confidence"

    out = df.copy()
    out["confidence_bucket"] = bucket
    rows = []
    for b in ("high_confidence", "low_confidence"):
        sub = out[out["confidence_bucket"] == b]
        if sub.empty:
            continue
        err = sub["karcher_pred"] - sub["actual_future_price"]
        pred_up = sub["karcher_pred"] > sub["current_price"]
        rows.append({
            "bucket": b,
            "MAE": err.abs().mean(),
            "directional_accuracy": float((pred_up == sub["actual_up"]).mean()),
            "n_folds": len(sub),
        })
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> pd.DataFrame:
    methods = {"naive": "naive_pred", "karcher": "karcher_pred", "karcher+ai_blend": "ai_blend_pred"}
    rows = []
    for name, col in methods.items():
        if col not in df.columns:
            continue
        err = df[col] - df["actual_future_price"]
        pred_up = df[col] > df["current_price"]
        rows.append({
            "method": name,
            "MAE": err.abs().mean(),
            "RMSE": float(np.sqrt((err**2).mean())),
            "directional_accuracy": float((pred_up == df["actual_up"]).mean()),
            "n_folds": int(df[col].notna().sum()),
        })
    if "dirflip_correct" in df.columns:
        rows.append({
            "method": "karcher+dirflip_gate",
            "MAE": df["karcher_pred"].sub(df["actual_future_price"]).abs().mean(),  # price level unchanged
            "RMSE": float(np.sqrt(((df["karcher_pred"] - df["actual_future_price"]) ** 2).mean())),
            "directional_accuracy": float(df["dirflip_correct"].mean()),
            "n_folds": int(df["dirflip_correct"].notna().sum()),
        })
    return pd.DataFrame(rows).sort_values("MAE")


def run_one(ticker: str, asset_name: str, timeframe: str) -> pd.DataFrame | None:
    freq = "D" if ticker in ("BTC-USD", "ETH-USD") else "B"
    horizon, n_folds, step_back, min_train = (3, 80, 2, 20) if timeframe == "daily" else (6, 120, 4, 25)
    try:
        price_df = download_prices(ticker, timeframe)
    except Exception as exc:
        print(f"  [data skip] {ticker} {timeframe}: {exc}")
        return None
    folds = collect_folds(price_df, ticker, asset_name, horizon, n_folds, step_back, freq)
    if len(folds) < min_train + 10:
        print(f"  [skip] {ticker} {timeframe}: only {len(folds)} usable folds")
        return None
    scored = run_nested_ensemble(folds, min_train_folds=min_train)
    result = score(scored)
    result.insert(0, "timeframe", timeframe)
    result.insert(0, "asset", asset_name)
    print(f"\n=== {asset_name} ({ticker}), {timeframe}, horizon={horizon} steps, "
          f"{len(scored)} scored folds (of {len(folds)} total) ===")
    print(result.to_string(index=False))

    gate = confidence_gate_breakdown(folds, min_train_folds=min_train)
    if not gate.empty:
        print(f"--- confidence-gate breakdown (karcher only, |wind_t_stat| vs. expanding median) ---")
        print(gate.to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None, help="single ticker, e.g. BTC-USD")
    parser.add_argument("--timeframe", default="daily", choices=["daily", "hourly", "both"])
    args = parser.parse_args()

    timeframes = ["daily", "hourly"] if args.timeframe == "both" else [args.timeframe]
    all_results = []
    for tf in timeframes:
        universe = DAILY_ASSETS if tf == "daily" else HOURLY_ASSETS
        if args.asset:
            universe = {args.asset: args.asset}
        for ticker, name in universe.items():
            r = run_one(ticker, name, tf)
            if r is not None:
                all_results.append(r)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        pd.set_option("display.float_format", lambda x: f"{x:.5f}")
        print("\n\n===== full scoreboard =====")
        print(combined.to_string(index=False))
