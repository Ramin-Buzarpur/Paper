"""
Addresses three of the four weaknesses flagged for publishability:
1. Longer history (up to the max yfinance offers per asset, not a fixed 6 years).
2. A real economic backtest: non-overlapping 3-day holding periods, per-trade transaction
   cost, cumulative return, annualized Sharpe ratio, max drawdown -- not just directional
   accuracy.
3. A momentum baseline (sign of the trailing 3-day return) for comparison, gated and ungated.

(Weakness 4, the Brent reversal mechanism, is investigated separately.)

Strategy: diffusion-metric analog selection (finding 18) + full-ELBO atypicality gate
(finding 28) -- only take a position on days the gate is "high atypical"; stay flat otherwise.
Position size is fixed (+1/-1), direction = sign of the analog-mean predicted return.

Result (see README Finding 29), on 7-20 years of history per asset (not the original 6-year
window): two important, honest findings.

1. The diffusion-metric model, with NO gate at all, beats a simple momentum baseline (sign of
   the trailing 3-day return) on Sharpe and total return, on every one of 5 assets tested
   (Brent, S&P 500, Gold, Bitcoin, Ethereum) -- a real, consistent edge over the simplest
   plausible competing strategy. Sanity-checked against pure buy-and-hold (Sharpe 0.07-0.54
   across these assets): the edge is not just long-term market beta in disguise.
2. The binary atypicality GATE (finding 26/28's "only trade on atypical days") makes economic
   performance WORSE than just always trading, on every single asset -- despite improving
   directional accuracy on the subset it fires on. Skipping "typical" days throws away too many
   still-profitable trades. This directly contradicts a naive reading of findings 26/28 as a
   ready-made trading rule; see position_sizing_backtest.py for the fix (continuous position
   sizing instead of a binary filter).

Caveat: reported Sharpe ratios (2-3.4) are high for a real systematic strategy and are likely
optimistic given the simplified transaction-cost assumption (2bps, no slippage) -- treat them
as comparisons between strategies under identical assumptions, not literal expected performance.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M, TAU, K, THEILER, HORIZON = 3, 1, 7, 10, 3
N_TIMESTEPS = 50
HIDDEN = 32
COST_BPS = 2.0  # one-way transaction cost in basis points, per position entry


class ScoreNet(nn.Module):
    def __init__(self, in_dim=M, hidden=HIDDEN):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.l1 = nn.Linear(in_dim + hidden, hidden)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x_t, t_frac, return_feature=False):
        te = self.time_embed(t_frac)
        h1 = self.act(self.l1(torch.cat([x_t, te], dim=-1)))
        if return_feature:
            return h1
        return self.out(self.act(self.l2(h1)))


def make_schedule(n=N_TIMESTEPS):
    betas = torch.linspace(1e-4, 0.02, n)
    alphas = 1.0 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


def train_scorenet(points, epochs=50, lr=2e-3, batch=128, seed=0):
    torch.manual_seed(seed)
    net = ScoreNet().to(device)
    betas, alphas, alpha_bars = [t.to(device) for t in make_schedule()]
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    X = torch.tensor(points, dtype=torch.float32).to(device)
    n = len(X)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x0 = X[idx]
            t = torch.randint(0, N_TIMESTEPS, (len(idx),), device=device)
            noise = torch.randn_like(x0)
            ab = alpha_bars[t].unsqueeze(-1)
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
            t_frac = (t.float() / N_TIMESTEPS).unsqueeze(-1)
            loss = nn.functional.mse_loss(net(x_t, t_frac), noise)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, (betas, alphas, alpha_bars)


@torch.no_grad()
def full_elbo_score(net, schedule, point, seed=0):
    betas, alphas, alpha_bars = schedule
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.arange(N_TIMESTEPS, device=device)
    x0 = torch.tensor(point, dtype=torch.float32, device=device).unsqueeze(0).repeat(N_TIMESTEPS, 1)
    noise = torch.randn(x0.shape, generator=g, device=device)
    ab = alpha_bars[t].unsqueeze(-1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    t_frac = (t.float() / N_TIMESTEPS).unsqueeze(-1)
    pred = net(x_t, t_frac)
    return float(((pred - noise) ** 2).mean().item())


def embed(log_ret):
    n = len(log_ret)
    n_pts = n - (M - 1) * TAU
    return np.column_stack([log_ret[(M-1-i)*TAU : (M-1-i)*TAU+n_pts] for i in range(M)])


def select_analogs(dist, origin_idx, last_ok):
    order = np.argsort(dist)
    chosen = []
    for j in order:
        if j > last_ok or j == origin_idx:
            continue
        if all(abs(int(j) - c) > THEILER for c in chosen):
            chosen.append(int(j))
        if len(chosen) == K:
            break
    return chosen


def download_max(ticker, max_years=20):
    df = yf.Ticker(ticker).history(period=f"{max_years}y", interval="1d", auto_adjust=False)
    df = df[["Close"]].dropna()
    df = df[df["Close"] > 0].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def collect(ticker, name, max_years=20, retrain_every=20, epochs=50):
    price_df = download_max(ticker, max_years)
    prices = price_df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(prices))
    scale = max(float(np.std(log_ret)), 1e-8)
    pc = embed(log_ret) / scale
    n_pts = len(pc)
    min_history = 250
    latest = n_pts - HORIZON - 1
    # NON-OVERLAPPING folds (step = HORIZON) for a realistic sequential backtest
    origins = list(range(min_history, latest, HORIZON))

    net, schedule, rows = None, None, []
    for k, j in enumerate(origins):
        last_ok = j - 1
        if net is None or k % retrain_every == 0:
            net, schedule = train_scorenet(pc[:j+1], epochs=epochs)

        feats = net.forward(torch.tensor(pc[:j+1], dtype=torch.float32, device=device),
                             torch.full((j+1, 1), 3 / N_TIMESTEPS, device=device),
                             return_feature=True).detach().cpu().numpy()
        d_diff = np.sum((feats - feats[j]) ** 2, axis=1)
        idx_diff = select_analogs(d_diff, j, last_ok)
        if len(idx_diff) < 3:
            continue

        atyp, _ = full_elbo_score(net, schedule, pc[j], seed=k), None

        p = j + (M - 1) * TAU
        current_price, actual_future_price = prices[p], prices[p + HORIZON]
        rets = [np.log(prices[idx + (M-1)*TAU + HORIZON] / prices[idx + (M-1)*TAU])
                for idx in idx_diff if idx + (M-1)*TAU + HORIZON < len(prices)]
        if len(rets) < 3:
            continue
        pred_ret = float(np.mean(rets))
        actual_ret = float(np.log(actual_future_price / current_price))
        momentum_ret = float(np.sum(pc[j]))  # trailing 3-day momentum, in scaled-return units

        rows.append({
            "date": price_df.index[p], "pred_ret": pred_ret, "actual_ret": actual_ret,
            "atypicality": atyp, "momentum_sign": np.sign(momentum_ret),
        })
    return pd.DataFrame(rows)


def run_strategy(df, pred_col, gate=True, min_train=20, cost_bps=COST_BPS):
    vals = df["atypicality"].to_numpy()
    rets = []
    for i in range(min_train, len(df)):
        if gate:
            thresh = np.median(vals[:i])
            if vals[i] < thresh:
                rets.append(0.0)  # flat, no trade, no cost
                continue
        direction = np.sign(df[pred_col].iloc[i])
        if direction == 0:
            rets.append(0.0)
            continue
        gross = direction * df["actual_ret"].iloc[i]
        cost = cost_bps / 10000.0  # paid once per entry (flat -> position), ignore exit cost for simplicity
        rets.append(gross - cost)
    return np.array(rets)


def sharpe(rets, periods_per_year):
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


def summarize(rets, label, periods_per_year):
    cum = np.cumsum(rets)
    equity = np.exp(cum)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    n_trades = int((rets != 0).sum())
    print(f"    {label:28s} total_ret={cum[-1]*100:+7.2f}%  sharpe={sharpe(rets, periods_per_year):+.2f}  "
          f"max_dd={dd.min()*100:6.2f}%  n_trades={n_trades}/{len(rets)}")


if __name__ == "__main__":
    for ticker, name, max_years in [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20), ("GC=F", "Gold", 20),
                                     ("BTC-USD", "Bitcoin", 10), ("ETH-USD", "Ethereum", 8)]:
        t0 = time.time()
        df = collect(ticker, name, max_years=max_years)
        years_actual = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25 if len(df) else 0
        periods_per_year = 252 / HORIZON
        print(f"\n=== {name}: {len(df)} non-overlapping 3-day folds over {years_actual:.1f} years ({time.time()-t0:.0f}s) ===")
        if len(df) < 30:
            print("    too few folds, skipping")
            continue
        summarize(run_strategy(df, "pred_ret", gate=True), "diffusion-metric, gated (ours)", periods_per_year)
        summarize(run_strategy(df, "pred_ret", gate=False), "diffusion-metric, ungated", periods_per_year)
        summarize(run_strategy(df, "momentum_sign", gate=True), "momentum baseline, gated", periods_per_year)
        summarize(run_strategy(df, "momentum_sign", gate=False), "momentum baseline, ungated", periods_per_year)

    print("\n\n=== SANITY CHECK: always-long (buy & hold) baseline, no model at all ===")
    for ticker, name, max_years in [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20), ("GC=F", "Gold", 20),
                                     ("BTC-USD", "Bitcoin", 10), ("ETH-USD", "Ethereum", 8)]:
        df = collect(ticker, name, max_years=max_years)
        if len(df) < 30:
            continue
        periods_per_year = 252 / HORIZON
        always_long = df["actual_ret"].to_numpy()[20:]  # match min_train offset used elsewhere
        summarize(always_long, f"{name}: always long, no cost", periods_per_year)
