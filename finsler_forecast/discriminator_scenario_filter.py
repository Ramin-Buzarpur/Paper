"""
AI-in-scenario-analysis test, motivated by DFMC-TimeGAN (Monte Carlo scenario generation
filtered by a GAN discriminator): train a small classifier to tell a real analog continuation
path apart from an incoherent one (here, a random bootstrap draw of individual historical daily
returns -- plausible pointwise, but not a coherent continuation), then use its "realism" score
to re-weight the K analogs before averaging, instead of the plain equal weight.

Unlike DFMC-TimeGAN, no new scenarios are generated -- the K analogs stay the true historical
continuations the colleague's method is built on (see geometry.py's own note on why a curved,
synthetic geodesic spray can't be honestly derived from this metric); the discriminator only
re-weights among them.

Result (see README Finding 7): the realism-weighted mean is statistically indistinguishable from
plain karcher on all three assets tested. The classifier evidently can't tell the two classes
apart well enough, on the simple path features used here (cumulative return, path vol, max
absolute step, autocorrelation), to produce weights that differ meaningfully from uniform.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import download_close_prices
from geometry import FinslerRandersGeodesicEngine

def path_features(returns_path):
    # returns_path: (horizon,) daily log returns of one scenario
    cum = returns_path.sum()
    vol = returns_path.std() if len(returns_path) > 1 else 0.0
    max_abs = np.abs(returns_path).max()
    autocorr = float(np.corrcoef(returns_path[:-1], returns_path[1:])[0, 1]) if len(returns_path) > 2 and returns_path.std() > 0 else 0.0
    return [cum, vol, max_abs, autocorr]

def collect(ticker, asset_name, horizon=3, n_scenarios=7, n_folds=80, step_back=2, freq="B", seed=0):
    rng = np.random.default_rng(seed)
    price_df = download_close_prices(ticker, 6)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret_full = np.diff(np.log(prices))
    dates = price_df.index
    n_total = len(prices)
    min_history = 200
    latest_origin = n_total - horizon - 1
    origins = sorted(sorted(range(latest_origin, min_history, -step_back)[:n_folds]))
    rows = []
    for origin in origins:
        current_price = prices[origin]
        actual_future_price = prices[origin + horizon]
        try:
            engine = (FinslerRandersGeodesicEngine(ticker=ticker, asset_name=asset_name, freq=freq,
                                                     drop_last_bar=False, n_scenarios=n_scenarios)
                      .fetch_and_reconstruct(price_df.iloc[:origin+1]).simulate(horizon))
        except Exception:
            continue
        if engine.k_used != n_scenarios:
            continue
        # real analog daily-return paths (from engine's own scenario state trajectories)
        analog_rets = np.diff(np.log(engine.scenario_prices), axis=1)  # (K, horizon)
        # negative class: random horizon-length windows drawn from full history up to origin (plausible
        # individual values, but NOT a coherent analog continuation -- the thing a discriminator should reject)
        pool = log_ret_full[:origin]
        fake_rets = np.array([rng.choice(pool, size=horizon, replace=True) for _ in range(n_scenarios)])

        feats_real = [path_features(r) for r in analog_rets]
        feats_fake = [path_features(r) for r in fake_rets]
        karcher_pred = float(engine.karcher_mean_price[-1])
        analog_log_ret_h = np.log(engine.scenario_prices[:, -1] / current_price)
        rows.append({
            "origin_date": dates[origin], "current_price": current_price,
            "actual_future_price": actual_future_price, "actual_up": actual_future_price > current_price,
            "karcher_pred": karcher_pred, "feats_real": feats_real, "feats_fake": feats_fake,
            "analog_log_ret_h": analog_log_ret_h,
        })
    return pd.DataFrame(rows)

def run_nested(df, min_train=20):
    scored = []
    for i in range(min_train, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i]
        X_train = np.array(sum(train["feats_real"].tolist(), []) + sum(train["feats_fake"].tolist(), []))
        y_train = np.array([1]*sum(len(f) for f in train["feats_real"]) + [0]*sum(len(f) for f in train["feats_fake"]))
        clf = LogisticRegression(max_iter=500, C=0.5).fit(X_train, y_train)

        test_feats = np.array(test["feats_real"])
        realism = clf.predict_proba(test_feats)[:, 1]  # P(this analog is a real, coherent continuation)
        weights = realism / realism.sum() if realism.sum() > 0 else np.ones_like(realism) / len(realism)
        filtered_ret = float(np.dot(weights, test["analog_log_ret_h"]))
        filtered_pred = test["current_price"] * np.exp(filtered_ret)

        scored.append({
            "actual_future_price": test["actual_future_price"], "current_price": test["current_price"],
            "actual_up": test["actual_future_price"] > test["current_price"],
            "karcher_pred": test["karcher_pred"], "filtered_pred": filtered_pred,
        })
    return pd.DataFrame(scored)

def score(df):
    out = []
    for col in ["karcher_pred", "filtered_pred"]:
        err = df[col] - df["actual_future_price"]
        up = df[col] > df["current_price"]
        out.append({"method": col, "MAE": err.abs().mean(), "dir_acc": float((up == df["actual_up"]).mean())})
    return pd.DataFrame(out)

for ticker, name, freq in [("BZ=F", "Brent", "B"), ("^GSPC", "S&P500", "B"), ("BTC-USD", "Bitcoin", "D")]:
    df = collect(ticker, name, freq=freq)
    scored = run_nested(df)
    print(f"\n=== {name}: {len(scored)} scored folds — discriminator-weighted analogs vs plain karcher ===")
    print(score(scored).to_string(index=False))
