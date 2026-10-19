"""
Downloads EUR/USD 1-minute bars from Dukascopy's public datafeed and prepares
hourly log-return series for diffusion training.

Why Dukascopy: Yahoo Finance's free intraday endpoint hard-caps hourly FX data
at ~730 days. Dukascopy publishes one 1-minute candle file per day going back
to the early 2000s, which is what a diffusion model actually needs.

File format (reverse-engineered, verified against real files):
    URL:    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YEAR}/{MONTH-1:02d}/{DAY:02d}/{SIDE}_candles_min_1.bi5
    Body:   LZMA-compressed, N x 24-byte big-endian records:
                int32 seconds_since_midnight, int32 open, int32 close,
                int32 low, int32 high, float32 volume
            price = raw_int / point_value (100000 for EURUSD, 1000 for JPY pairs)
    Missing days (weekends, holidays, before the instrument's start) return an
    empty body — that's normal, not an error.

Public HTTPS endpoint above is the verified, no-auth way to get this data. (A
"historical-data-service" page once pointed at an S3 bucket with "Requester
Pays" billing under a name unrelated to Dukascopy — that bills whoever runs
the request, a classic trap. The endpoint above was verified independently
with plain curl: no AWS, no account, no payment.)

Usage:
    python -m finsler_forecast.diffusion.dataset                # full download
    python -m finsler_forecast.diffusion.dataset --probe-only   # just find the
                                                                   first available day
"""

from __future__ import annotations

import lzma
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from .denoise import remove_flat_candles

SYMBOL = "EURUSD"
SIDE = "BID"
POINT_VALUE = 100_000  # 1000 for JPY pairs

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
RAW_DAY_CACHE = os.path.join(DATA_DIR, "dukascopy_days")
MERGED_PARQUET = os.path.join(DATA_DIR, f"{SYMBOL}_minute.parquet")

URL_TEMPLATE = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{side}_candles_min_1.bi5"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
RECORD_SIZE = 24  # 5 x int32 + 1 x float32

MAX_RETRIES = 5
BASE_BACKOFF = 3.0
MAX_THREADS = 8


def _cache_path(day: date) -> str:
    return os.path.join(RAW_DAY_CACHE, f"{day.isoformat()}.bi5")


def fetch_raw_day(day: date, session: requests.Session) -> bytes | None:
    """Raw .bi5 bytes for a day, or None if no data exists. Cached on disk so reruns resume."""
    cache_path = _cache_path(day)
    if os.path.exists(cache_path):
        content = open(cache_path, "rb").read()
        return content if content else None

    url = URL_TEMPLATE.format(symbol=SYMBOL, year=day.year, month=day.month - 1, day=day.day, side=SIDE)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(BASE_BACKOFF * attempt)
            continue
        if resp.status_code == 200:
            content = bytes(resp.content)
            # 0-byte cache file marks a known-empty day (weekend/holiday), so reruns skip it too
            with open(cache_path, "wb") as f:
                f.write(content)
            return content if content else None
        if resp.status_code in (429, 503):
            time.sleep(BASE_BACKOFF * attempt)
            continue
        return None  # 404 or similar: no data that day
    return None


def decode_day(raw: bytes, day: date) -> pd.DataFrame:
    decompressed = lzma.decompress(raw)
    n = len(decompressed) // RECORD_SIZE
    flat = struct.unpack(">" + "iiiiif" * n, decompressed[: n * RECORD_SIZE])
    arr = np.array(flat, dtype=np.float64).reshape(n, 6)
    seconds, o, c, lo, hi, vol = (arr[:, i] for i in range(6))
    timestamps = pd.Timestamp(day) + pd.to_timedelta(seconds, unit="s")
    return pd.DataFrame(
        {
            "Datetime": timestamps,
            "Open": o / POINT_VALUE,
            "High": hi / POINT_VALUE,
            "Low": lo / POINT_VALUE,
            "Close": c / POINT_VALUE,
            "Volume": vol,
        }
    )


def find_first_available_day(probe_start: date = date(2000, 1, 3)) -> date:
    """Binary search for the earliest day Dukascopy has data for."""
    session = requests.Session()

    def has_data(d: date) -> bool:
        if d.weekday() == 5:  # Saturday never has data; nudge to Friday
            d -= timedelta(days=1)
        found = fetch_raw_day(d, session) is not None
        time.sleep(0.5)  # be polite between probes
        return found

    lo, hi = probe_start, date.today() - timedelta(days=3)
    if has_data(lo):
        return lo
    if not has_data(hi):
        raise RuntimeError("could not confirm any data exists near today; check connectivity")

    lo_ord, hi_ord = lo.toordinal(), hi.toordinal()
    while hi_ord - lo_ord > 1:
        mid_ord = (lo_ord + hi_ord) // 2
        if has_data(date.fromordinal(mid_ord)):
            hi_ord = mid_ord
        else:
            lo_ord = mid_ord
    return date.fromordinal(hi_ord)


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() != 5:  # FX market closed Saturdays; Dukascopy never has Saturday files
            yield d
        d += timedelta(days=1)


def download_range(first_day: date, last_day: date, threads: int = MAX_THREADS) -> None:
    os.makedirs(RAW_DAY_CACHE, exist_ok=True)
    days = list(_weekdays(first_day, last_day))
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(fetch_raw_day, d, session) for d in days]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 250 == 0:
                print(f"  {i}/{len(days)} days processed")


def merge_cached_days(first_day: date, last_day: date) -> pd.DataFrame:
    frames = []
    for d in _weekdays(first_day, last_day):
        path = _cache_path(d)
        if not os.path.exists(path):
            continue
        raw = open(path, "rb").read()
        if raw:
            frames.append(decode_day(raw, d))
    if not frames:
        raise RuntimeError("no decodable days found; run download_range first")
    df = pd.concat(frames, ignore_index=True).sort_values("Datetime")
    return df.drop_duplicates(subset="Datetime").reset_index(drop=True)


def build_dataset(probe_start: date = date(2000, 1, 3), probe_only: bool = False) -> date:
    os.makedirs(DATA_DIR, exist_ok=True)
    print("finding first available day on Dukascopy...")
    first_day = find_first_available_day(probe_start)
    print(f"first available day: {first_day}")
    if probe_only:
        return first_day

    last_day = date.today() - timedelta(days=1)  # today's bar is likely incomplete
    print(f"downloading {SYMBOL} minute data from {first_day} to {last_day}...")
    download_range(first_day, last_day)
    print("merging cached days...")
    df = merge_cached_days(first_day, last_day)
    df.to_parquet(MERGED_PARQUET, index=False)
    print(f"saved {len(df)} minute bars to {MERGED_PARQUET}")
    return first_day


def resample_to_hourly(minute_df: pd.DataFrame) -> pd.DataFrame:
    # drop closed-market candles first, so an hour that's entirely a market closure
    # (e.g. the weekend) has no rows left and resamples to NaN instead of a fake
    # zero-volatility bar at the last pre-closure price
    clean_df = remove_flat_candles(minute_df)
    hourly = clean_df.set_index("Datetime")["Close"].resample("1h").last().dropna()
    return hourly.to_frame("Close")


def hourly_log_return_series(parquet_path: str = MERGED_PARQUET) -> pd.Series:
    """Hourly log returns indexed by timestamp, for date-aligned backtesting."""
    minute_df = pd.read_parquet(parquet_path)
    hourly_close = resample_to_hourly(minute_df)["Close"]
    return np.log(hourly_close).diff().dropna()


def load_hourly_log_returns(parquet_path: str = MERGED_PARQUET) -> tuple[np.ndarray, float]:
    """Plain array + bars/day, for training (where timestamps don't matter)."""
    series = hourly_log_return_series(parquet_path)
    bars_per_day = float(series.groupby(series.index.date).size().median())
    return series.to_numpy(dtype=np.float64), bars_per_day


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    build_dataset(probe_only=args.probe_only)
