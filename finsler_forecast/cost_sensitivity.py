"""
Transaction-cost sensitivity for the position_sizing_backtest.py strategy (sized + momentum
filter): is the 2bps cost assumption used everywhere else load-bearing? Sweeps 2/10/25/50bps.

Result (see README Finding 30): the strategy is robust to cost. Even at 50bps (25x the base
assumption, already conservative for liquid futures/equities and plausible for crypto), Sharpe
drops only modestly: Brent 3.22->2.89, S&P 500 2.69->1.96, Gold 3.08->2.23, Bitcoin 3.57->3.27,
Ethereum 3.14->2.90. The 3-day holding period means turnover is low enough that cost assumptions
don't drive the headline result -- this removes one of the easiest objections a reviewer could
raise.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from economic_backtest import collect, sharpe, summarize, HORIZON
from position_sizing_backtest import run_sized_strategy


def main():
    for ticker, name, max_years in [("BZ=F", "Brent", 20), ("^GSPC", "S&P500", 20), ("GC=F", "Gold", 20),
                                     ("BTC-USD", "Bitcoin", 10), ("ETH-USD", "Ethereum", 8)]:
        t0 = time.time()
        df = collect(ticker, name, max_years=max_years)
        if len(df) < 30:
            continue
        periods_per_year = 252 / HORIZON
        print(f"\n=== {name} ({time.time()-t0:.0f}s) ===")
        for cost_bps in (2.0, 10.0, 25.0, 50.0):
            rets = run_sized_strategy(df, "pred_ret", momentum_filter=True, cost_bps=cost_bps)
            summarize(rets, f"cost={cost_bps:.0f}bps", periods_per_year)


if __name__ == "__main__":
    main()
