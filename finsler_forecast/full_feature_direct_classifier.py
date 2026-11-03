"""
Last direct-prediction attempt: a gradient-boosted classifier using every engineered feature
found useful anywhere in this directory at once (wind strength/t-stat, neighbour anisotropy,
analog spread, momentum/vol, volume z-score, bootstrap spread) to directly classify tomorrow's
direction -- not correct or gate karcher, replace it outright when the classifier beats karcher
on a held-out validation split (falls back to karcher otherwise, so it can only help, never by
construction look worse than the ungated version on the folds it chooses not to touch).

Result (see README Finding 17): still loses. Even with the richest feature set assembled in
this whole investigation and an explicit fallback-to-karcher safety net, the classifier's
chosen folds perform worse than karcher alone on S&P 500 (43.6% vs. 49.1%) and Bitcoin (40.0%
vs. 52.7%), and tie on Brent (61.8% vs 61.8%, classifier used in only 15% of folds there). This
is the clearest demonstration yet that the bottleneck across every one of the ~15 direct-
prediction attempts in this directory is not feature richness or model choice -- it's that
tomorrow's direction, 3 days out, is close to unlearnable from any combination of these
features, for these assets, at this sample size. What has worked (findings 6, 8, 13, 14) has
never been "predict better" -- it's been "decide when to trust the existing forecast" or
"change what counts as a similar historical day."
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from geometry import FinslerRandersGeodesicEngine
from ensemble import engine_meta_features

rng = np.random.default_rng(0)

def download_with_volume(ticker, years):
    df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=False)
    df = df[["Close", "Volume"]].dropna(); df = df[df["Close"] > 0].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()

FEATS = ["wind_strength", "wind_t_stat", "neighbour_anisotropy", "analog_spread", "analog_mean_ret",
         "recent_vol_5", "recent_vol_21", "momentum_5", "momentum_21", "vol_z", "boot_spread"]

def collect(ticker, name, horizon=3, n_scenarios=7, n_folds=80, step_back=2, n_boot=200):
    df = download_with_volume(ticker, 6)
    prices = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    vol_roll_mean = pd.Series(volume).rolling(20, min_periods=5).mean().to_numpy()
    vol_roll_std = pd.Series(volume).rolling(20, min_periods=5).std().to_numpy()
    vol_z = (volume - vol_roll_mean) / np.where(vol_roll_std > 0, vol_roll_std, 1.0)
    n_total = len(prices); min_history, latest = 200, n_total - horizon - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))
    rows = []
    for origin in origins:
        current_price = prices[origin]; actual = prices[origin + horizon]
        try:
            engine = (FinslerRandersGeodesicEngine(ticker=ticker, asset_name=name, drop_last_bar=False, n_scenarios=n_scenarios)
                      .fetch_and_reconstruct(df[["Close"]].iloc[:origin+1]).simulate(horizon))
        except Exception:
            continue
        if engine.k_used != n_scenarios: continue
        analog_log_ret = np.log(engine.scenario_prices[:, -1] / current_price)
        boot_spread = float(rng.choice(analog_log_ret, size=(n_boot, n_scenarios), replace=True).mean(axis=1).std())
        feats = engine_meta_features(engine)
        feats["vol_z"] = vol_z[origin]; feats["boot_spread"] = boot_spread
        row = {"current_price": current_price, "actual_future_price": actual,
               "actual_up": int(actual > current_price), "karcher_pred": float(engine.karcher_mean_price[-1])}
        row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows).dropna().reset_index(drop=True)

def run_nested(df, min_train=25, val_frac=0.3):
    scored = []
    for i in range(min_train, len(df)):
        train, test = df.iloc[:i], df.iloc[i]
        n_val = max(5, int(len(train)*val_frac))
        fit_part, val_part = train.iloc[:-n_val], train.iloc[-n_val:]
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.06, min_samples_leaf=10)
        clf.fit(fit_part[FEATS], fit_part["actual_up"])
        val_acc = (clf.predict(val_part[FEATS]) == val_part["actual_up"]).mean()
        karcher_val_acc = ((val_part["karcher_pred"] > val_part["current_price"]).astype(int) == val_part["actual_up"]).mean()
        use_clf = val_acc > karcher_val_acc  # only use the classifier if it beat karcher on held-out validation
        clf_full = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.06, min_samples_leaf=10)
        clf_full.fit(train[FEATS], train["actual_up"])
        pred_up = int(clf_full.predict(test[FEATS].to_numpy().reshape(1,-1))[0]) if use_clf else int(test["karcher_pred"] > test["current_price"])
        scored.append({"actual_up": test["actual_up"], "pred_up": pred_up,
                        "karcher_up": int(test["karcher_pred"] > test["current_price"]), "used_clf": use_clf})
    return pd.DataFrame(scored)

for ticker, name in [("^GSPC","S&P500"), ("BZ=F","Brent"), ("BTC-USD","Bitcoin")]:
    df = collect(ticker, name)
    scored = run_nested(df)
    acc_clf_gated = (scored["pred_up"] == scored["actual_up"]).mean()
    acc_karcher = (scored["karcher_up"] == scored["actual_up"]).mean()
    print(f"{name:10s} n={len(scored):3d}  gated-classifier dir_acc={acc_clf_gated:.3f}  "
          f"karcher-alone dir_acc={acc_karcher:.3f}  classifier used in {scored['used_clf'].mean():.2f} of folds")
