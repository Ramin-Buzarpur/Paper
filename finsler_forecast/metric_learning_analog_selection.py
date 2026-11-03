"""
Distance-metric-learning test for analog SELECTION (not reweighting/filtering the K analogs
after they're picked, which is what ensemble.py/discriminator_scenario_filter.py already tried
and failed -- this changes which K neighbours get picked in the first place).

Mechanism: the engine's embedding is x_j = [r(t), r(t-tau), r(t-2tau)]. Instead of the engine's
local Ledoit-Wolf-covariance Mahalanobis metric, scale the three coordinates by a learned
per-coordinate weight vector before a plain Euclidean kNN search, i.e. learn how much each lag
should count toward "this historical state is similar to today." The weight is picked causally
(nested validation on past folds only, from a small fixed grid) per fold, never on the fold it
scores -- same discipline as every other experiment in this directory.

Result (see README Finding 8): hurts Brent (CPCV: loses to the unweighted default in 64% of
28 combinatorial fold-groupings) and is a coin flip on S&P 500 (50%), but is a robust,
CPCV-confirmed win on Bitcoin -- the one asset whose base karcher edge was already known (see
cpcv_robustness.py) to NOT survive CPCV. The learned weight consistently down-weights the
oldest lag r(t-2tau) (chosen (1.0, 1.0, 0.5) in 43/60 folds), and with that weighting, Bitcoin's
directional accuracy beats the unweighted default in 100% of 28 CPCV combinations (median 64.3%
vs 58.1%). This is the first improvement found so far that is specific to -- and actually
rescues -- the one asset the base method didn't already work on, rather than helping/hurting
assets where the base method already had (or already lacked) a verified edge.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from data import download_close_prices

M = 3
TAU = 1
K = 7
THEILER = 10

def embed(log_ret):
    n = len(log_ret)
    n_pts = n - (M - 1) * TAU
    pc = np.column_stack([log_ret[(M-1-i)*TAU : (M-1-i)*TAU+n_pts] for i in range(M)])
    return pc

def analog_pred(pc, prices, origin_idx_in_pc, weights, horizon):
    """weights scale the 3 embedding coords before Euclidean kNN; weights=[1,1,1] ~ unweighted."""
    n_pts = len(pc)
    last_ok = n_pts - 1 - horizon
    scaled = pc * weights
    tree = cKDTree(scaled[: last_ok + 1])
    x = scaled[origin_idx_in_pc]
    _, idx = tree.query(x, k=min(last_ok + 1, 10 * K + THEILER))
    idx = np.atleast_1d(idx)
    chosen = []
    for j in idx:
        j = int(j)
        if j == origin_idx_in_pc:
            continue
        if all(abs(j - c) > THEILER for c in chosen):
            chosen.append(j)
        if len(chosen) == K:
            break
    if len(chosen) < 3:
        return None
    # each analog's horizon-ahead return, measured from the ORIGINAL (unscaled) point cloud's col 0
    rets = np.array([pc[j + horizon, 0] if j + horizon < len(pc) else np.nan for j in chosen])
    # actually we need cumulative return over `horizon` days from analog state j:
    rets = np.array([np.sum(pc[j+1:j+horizon+1, 0]) if j + horizon < len(pc) else np.nan for j in chosen])
    rets = rets[~np.isnan(rets)]
    if len(rets) < 3:
        return None
    return float(rets.mean())

def collect(ticker, name, horizon=3, n_folds=80, step_back=2, weight_grid=None):
    price_df = download_close_prices(ticker, 6)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    dates = price_df.index
    log_ret = np.diff(np.log(prices))
    pc = embed(log_ret)
    scale = max(float(np.std(log_ret)), 1e-12)
    n_pts = len(pc)
    min_history = 200
    latest_origin_pc = n_pts - horizon - 1
    origins_pc = sorted(sorted(range(latest_origin_pc, min_history, -step_back)[:n_folds]))

    rows = []
    for j in origins_pc:
        price_idx = j + (M - 1) * TAU  # index into `prices` for this embedded point's "current" bar
        current_price = prices[price_idx]
        actual_future_price = prices[price_idx + horizon]
        preds = {}
        for w in weight_grid:
            r = analog_pred(pc[: j + horizon + 1], prices, j, np.array(w), horizon)
            preds[w] = r
        rows.append({
            "current_price": current_price, "actual_future_price": actual_future_price,
            "actual_up": actual_future_price > current_price, "preds": preds,
        })
    return pd.DataFrame(rows)

def nested_pick_weight(df, weight_grid, min_train=20, val_frac=0.3):
    scored = []
    for i in range(min_train, len(df)):
        train, test = df.iloc[:i], df.iloc[i]
        n_val = max(5, int(len(train) * val_frac))
        val_part = train.iloc[-n_val:]
        best_w, best_mae = weight_grid[0], None
        for w in weight_grid:
            preds = val_part["preds"].apply(lambda d: d.get(w))
            valid = preds.notna()
            if valid.sum() < 5:
                continue
            err = (val_part["current_price"][valid] * np.exp(preds[valid])) - val_part["actual_future_price"][valid]
            mae = float(err.abs().mean())
            if best_mae is None or mae < best_mae:
                best_mae, best_w = mae, w
        r = test["preds"].get(best_w)
        default_r = test["preds"].get((1.0, 1.0, 1.0))
        if r is None or default_r is None:
            continue
        pred_price = test["current_price"] * np.exp(r)
        default_price = test["current_price"] * np.exp(default_r)
        scored.append({
            "actual_future_price": test["actual_future_price"], "current_price": test["current_price"],
            "actual_up": test["actual_up"], "learned_pred": pred_price, "default_pred": default_price,
            "chosen_weight": best_w,
        })
    return pd.DataFrame(scored)

def score(df):
    out = []
    for col in ["default_pred", "learned_pred"]:
        err = df[col] - df["actual_future_price"]
        up = df[col] > df["current_price"]
        out.append({"method": col, "MAE": err.abs().mean(), "dir_acc": float((up == df["actual_up"]).mean())})
    return pd.DataFrame(out)

WEIGHT_GRID = [(1.0,1.0,1.0), (1.0,0.5,0.25), (2.0,1.0,0.5), (1.0,1.0,0.5), (1.5,1.0,1.0), (0.5,1.0,1.0)]

for ticker, name in [("BZ=F", "Brent"), ("^GSPC", "S&P500"), ("BTC-USD", "Bitcoin")]:
    df = collect(ticker, name, weight_grid=WEIGHT_GRID)
    scored = nested_pick_weight(df, WEIGHT_GRID)
    print(f"\n=== {name}: {len(scored)} folds, metric-learned (per-coordinate weight) vs default(1,1,1) ===")
    print(score(scored).to_string(index=False))
    print("chosen weights distribution:", scored["chosen_weight"].value_counts().to_dict())

print("\n\n=== CPCV robustness check: BTC metric-learning result ===")
import itertools
def cpcv_on_scored(scored, n_blocks=8, k_test=2):
    n = len(scored)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[b], edges[b+1]) for b in range(n_blocks)]
    rows = []
    for test_blocks in itertools.combinations(range(n_blocks), k_test):
        idx = np.concatenate([blocks[b] for b in test_blocks])
        sub = scored.iloc[idx]
        if len(sub) < 5: continue
        for col in ("default_pred", "learned_pred"):
            err = sub[col] - sub["actual_future_price"]
            up = sub[col] > sub["current_price"]
            rows.append({"combo": test_blocks, "method": col,
                         "MAE": err.abs().mean(), "dir_acc": float((up == sub["actual_up"]).mean())})
    return pd.DataFrame(rows)

df_btc = collect("BTC-USD", "Bitcoin", weight_grid=WEIGHT_GRID)
scored_btc = nested_pick_weight(df_btc, WEIGHT_GRID)
cpcv = cpcv_on_scored(scored_btc)
d = cpcv[cpcv.method=="default_pred"].set_index("combo")["dir_acc"]
l = cpcv[cpcv.method=="learned_pred"].set_index("combo")["dir_acc"]
print(f"default median dir_acc={d.median():.3f}  learned median dir_acc={l.median():.3f}")
print(f"learned loses to default in {float((l < d).mean()):.2f} of {len(d)} combos")

print("\n=== CPCV check: Brent & S&P500 metric-learning ===")
for ticker, name in [("BZ=F","Brent"), ("^GSPC","S&P500")]:
    dfx = collect(ticker, name, weight_grid=WEIGHT_GRID)
    sx = nested_pick_weight(dfx, WEIGHT_GRID)
    cx = cpcv_on_scored(sx)
    d = cx[cx.method=="default_pred"].set_index("combo")["dir_acc"]
    l = cx[cx.method=="learned_pred"].set_index("combo")["dir_acc"]
    print(f"{name}: default median={d.median():.3f}  learned median={l.median():.3f}  learned loses in {float((l<d).mean()):.2f} of {len(d)}")
