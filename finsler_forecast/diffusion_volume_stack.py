"""
Does the diffusion-score similarity metric (finding 18, robust win on S&P 500) stack with the
volume-surprise gate (finding 14, universal)? Gates the diffusion-metric's own prediction by
causal volume-surprise, CPCV-checked. Also sweeps K (number of analogs) for the diffusion
metric on S&P 500 as a secondary, single-pass check (not yet nested-validated -- a lead, not a
finding).

Result (see README Finding 20): yes, they stack. CPCV-confirmed on S&P 500: diffusion-metric
alone scores 65.0% (median); gated to the high-volume-surprise subset, 70.0% (median, 28
combos); the low-volume subset drops to 58.6%. This is the best validated S&P 500 number in the
whole investigation. The K-sweep (K=15 scored 68.8% vs K=7's 66.2% in a single pass) hints at a
further tuning opportunity but needs the same nested-validation discipline before it can be
trusted -- flagged as a lead for future work, not adopted here.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import yfinance as yf
import diffusion_score_metric as dsm  # noqa: triggers its own __main__ prints, harmless

def download_with_volume(ticker, years):
    df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=False)
    df = df[["Close", "Volume"]].dropna(); df = df[df["Close"] > 0].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def collect(ticker, name, k_analogs, n_folds=80, step_back=2, retrain_every=15, epochs=50):
    df = download_with_volume(ticker, 6)
    prices = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    vol_mean = pd.Series(volume).rolling(20, min_periods=5).mean().to_numpy()
    vol_std = pd.Series(volume).rolling(20, min_periods=5).std().to_numpy()
    vol_z = (volume - vol_mean) / np.where(vol_std > 0, vol_std, 1.0)

    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = dsm.embed(log_ret) / scale
    n_pts = len(pc)
    M, TAU, HORIZON = dsm.M, dsm.TAU, dsm.HORIZON
    min_history, latest = 200, n_pts - HORIZON - 1
    origins = sorted(sorted(range(latest, min_history, -step_back)[:n_folds]))

    def select_analogs_k(dist, origin_idx, last_ok, k):
        order = np.argsort(dist)
        chosen = []
        for j in order:
            if j > last_ok or j == origin_idx:
                continue
            if all(abs(int(j) - c) > dsm.THEILER for c in chosen):
                chosen.append(int(j))
            if len(chosen) == k:
                break
        return chosen

    net, rows = None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
        if net is None or k % retrain_every == 0:
            net = dsm.train_scorenet(pc[:j+1], epochs=epochs)
        feats = dsm.diffusion_features(net, pc[:j+1])
        d_diff = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_diff = select_analogs_k(d_diff, j, last_ok, k_analogs)
        if len(idx_diff) < 3:
            continue
        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                for idx in idx_diff if idx + (M-1)*TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred = current_price * np.exp(np.mean(rets))
        rows.append({"current_price": current_price, "actual_future_price": actual_future_price,
                      "actual_up": actual_future_price > current_price, "diff_pred": pred,
                      "vol_z": vol_z[p]})
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def causal_bucket(df, min_train=20):
    vals = df["vol_z"].to_numpy()
    bucket = np.full(len(df), "warmup", dtype=object)
    for i in range(min_train, len(df)):
        bucket[i] = "high" if vals[i] >= np.median(vals[:i]) else "low"
    return bucket


def cpcv_stack(df, n_blocks=8, k_test=2, min_train=20):
    bucket = causal_bucket(df, min_train)
    work = df.iloc[min_train:].copy()
    work["bucket"] = bucket[min_train:]
    n = len(work)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[b], edges[b+1]) for b in range(n_blocks)]
    rows = []
    for combo in itertools.combinations(range(n_blocks), k_test):
        idx = np.concatenate([blocks[b] for b in combo])
        sub = work.iloc[idx]
        for b in ("high", "low"):
            bsub = sub[sub.bucket == b]
            if len(bsub) < 4:
                continue
            up = bsub["diff_pred"] > bsub["current_price"]
            rows.append({"bucket": b, "dir_acc": float((up == bsub["actual_up"]).mean()), "n": len(bsub)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("\n=== Step 1: CPCV-confirm diffusion-metric + volume gate stack (S&P 500, K=7) ===")
    df7 = collect("^GSPC", "S&P 500", k_analogs=7)
    c7 = cpcv_stack(df7)
    h7 = c7[c7.bucket == "high"]["dir_acc"]; l7 = c7[c7.bucket == "low"]["dir_acc"]
    print(f"high-volume median={h7.median():.3f} (n_combos={len(h7)})  low-volume median={l7.median():.3f} (n_combos={len(l7)})")

    print("\n=== Step 2: sweep K (number of analogs) for the diffusion-metric, S&P 500 ===")
    for k_val in (5, 7, 10, 15, 20):
        dfk = collect("^GSPC", "S&P 500", k_analogs=k_val)
        up = dfk["diff_pred"] > dfk["current_price"]
        acc = float((up == dfk["actual_up"]).mean())
        print(f"K={k_val:2d}  n={len(dfk):3d}  dir_acc={acc:.3f}")
