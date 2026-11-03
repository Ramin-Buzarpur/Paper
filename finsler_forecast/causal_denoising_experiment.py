"""
Preprocessing test, per the user's suggestion: denoise/normalize the signal before analog
matching. Implementation: a CAUSAL EMA smooth (span picked by nested validation on past folds)
is applied to the log-returns used ONLY for building the delay embedding / neighbour search --
the actual forecast target still averages each chosen analog's TRUE, undenoised future return,
so this never corrupts the thing being predicted, only what counts as "similar."

Standard wavelet/Fourier denoising is NOT used here because it's typically computed over the
whole series at once, which leaks future information into past points -- a real risk this
whole investigation has been built around avoiding. A causal (one-sided) EMA has no such leak.

Result (see README Finding 13), and a cautionary note on methodology: a naive single-path test
(pick the best EMA span by eyeballing results over the whole backtest) made it LOOK like
denoising helped S&P 500 (48.7% -> 56.2% dir. acc.). Redone properly -- span chosen by nested
validation on past folds only, then CPCV-checked -- that gain vanished entirely (51.7% either
way). This is exactly the kind of apparent improvement the nested/CPCV discipline in this
directory exists to catch.

| asset | no denoising | denoised (nested span) | CPCV: denoised loses in |
|---|---|---|---|
| S&P 500 | 51.7% | 51.7% (no real change) | 43% of combos (a wash) |
| Bitcoin | 58.1% | **64.6%** | 32% of combos (a real, moderate win) |
| Brent | 53.3% | 41.4% | 79% of combos (hurts) |

Denoising is a third thing (after bootstrap-gating and metric-learned analog selection) that
specifically helps Bitcoin and not the other two assets -- consistent with Bitcoin being the
noisiest of the three return series tested.
"""

import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from data import download_close_prices

M, TAU, K, THEILER, HORIZON = 3, 1, 7, 10, 3
SPAN_GRID = [None, 2, 3, 5, 8]

def causal_ema(x, span):
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    return out

def embed(series):
    n = len(series); n_pts = n - (M - 1) * TAU
    return np.column_stack([series[(M-1-i)*TAU : (M-1-i)*TAU+n_pts] for i in range(M)])

def analog_mean_return(pc_embed, raw_ret, origin_idx, horizon):
    n_pts = len(pc_embed); last_ok = n_pts - 1 - horizon
    tree = cKDTree(pc_embed[: last_ok + 1]); x = pc_embed[origin_idx]
    _, idx = tree.query(x, k=min(last_ok + 1, 10 * K + THEILER))
    idx = np.atleast_1d(idx); chosen = []
    for j in idx:
        j = int(j)
        if j == origin_idx: continue
        if all(abs(j - c) > THEILER for c in chosen): chosen.append(j)
        if len(chosen) == K: break
    if len(chosen) < 3: return None
    rets = []
    for j in chosen:
        p = j + (M - 1) * TAU
        if p + horizon < len(raw_ret): rets.append(raw_ret[p+1:p+1+horizon].sum())
    return float(np.mean(rets)) if len(rets) >= 3 else None

def collect_all_spans(ticker, name, n_folds=80, step_back=2):
    price_df = download_close_prices(ticker, 6)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    embed_series_by_span = {span: (causal_ema(log_ret, span) if span else log_ret) for span in SPAN_GRID}
    pc_by_span = {span: embed(s) for span, s in embed_series_by_span.items()}
    n_pts = len(pc_by_span[None]); min_history, latest = 200, n_pts - HORIZON - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))
    rows = []
    for j in origins:
        p = j + (M - 1) * TAU
        current_price, actual = prices[p], prices[p + HORIZON]
        preds = {}
        for span in SPAN_GRID:
            r = analog_mean_return(pc_by_span[span][: j + HORIZON + 1], log_ret, j, HORIZON)
            preds[span] = current_price * np.exp(r) if r is not None else None
        if preds[None] is None: continue
        rows.append({"current_price": current_price, "actual_future_price": actual,
                      "actual_up": actual > current_price, "preds": preds})
    return pd.DataFrame(rows)

def nested_pick(df, min_train=20, val_frac=0.3):
    scored = []
    for i in range(min_train, len(df)):
        train, test = df.iloc[:i], df.iloc[i]
        n_val = max(5, int(len(train)*val_frac)); val = train.iloc[-n_val:]
        best_span, best_mae = None, None
        for span in SPAN_GRID:
            p = val["preds"].apply(lambda d: d.get(span))
            valid = p.notna()
            if valid.sum() < 5: continue
            err = (p[valid] - val["actual_future_price"][valid]).abs().mean()
            if best_mae is None or err < best_mae: best_mae, best_span = err, span
        learned = test["preds"].get(best_span)
        default = test["preds"].get(None)
        if learned is None or default is None: continue
        scored.append({"actual_future_price": test["actual_future_price"], "current_price": test["current_price"],
                        "actual_up": test["actual_up"], "learned_pred": learned, "default_pred": default,
                        "chosen_span": best_span})
    return pd.DataFrame(scored)

def cpcv(scored, n_blocks=8, k_test=2):
    n = len(scored); edges = np.linspace(0, n, n_blocks+1).astype(int)
    blocks = [np.arange(edges[b], edges[b+1]) for b in range(n_blocks)]
    rows = []
    for combo in itertools.combinations(range(n_blocks), k_test):
        idx = np.concatenate([blocks[b] for b in combo]); sub = scored.iloc[idx]
        if len(sub) < 5: continue
        for col in ("default_pred","learned_pred"):
            err = sub[col]-sub["actual_future_price"]; up = sub[col]>sub["current_price"]
            rows.append({"combo":combo,"method":col,"MAE":err.abs().mean(),"dir_acc":float((up==sub["actual_up"]).mean())})
    return pd.DataFrame(rows)

for ticker, name in [("^GSPC","S&P500"), ("BTC-USD","Bitcoin"), ("BZ=F","Brent")]:
    df = collect_all_spans(ticker, name)
    scored = nested_pick(df)
    c = cpcv(scored)
    d = c[c.method=="default_pred"].set_index("combo")["dir_acc"]
    l = c[c.method=="learned_pred"].set_index("combo")["dir_acc"]
    print(f"{name:10s} n={len(scored):3d}  default median={d.median():.3f}  denoised median={l.median():.3f}  "
          f"denoised loses in {float((l<d).mean()):.2f} of {len(d)}  "
          f"span picks={scored['chosen_span'].value_counts().to_dict()}")
