from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_close_prices(ticker: str, years: int) -> pd.DataFrame:
    """Daily closes for `ticker`: tz-naive index, sorted, de-duplicated, strictly positive."""
    period = f"{int(years)}y"
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        if df is not None and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"no price data returned for {ticker}")

    df = df[["Close"]].dropna()
    df = df[df["Close"] > 0].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()
