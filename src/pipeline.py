from __future__ import annotations

import os

import numpy as np

from .config import Config
from .data.features import build_features
from .data.market_data import load_market_data
from .data.news import daily_sentiment_array, fetch_and_cache_news


def prepare_data(cfg: Config, fetch_news: bool = True, log=print):
    dcfg = cfg.section("data")
    ncfg = cfg.section("news")
    raw_dir_abs = os.path.normpath(dcfg.get("raw_dir", "./data/raw"))
    cache_dir_abs = os.path.normpath(dcfg.get("cache_dir", "./data_cache"))

    dates, symbols, arrays = load_market_data(
        raw_dir=raw_dir_abs,
        cache_dir=cache_dir_abs,
        markets=list(dcfg.get("markets", [])),
        max_stocks=int(dcfg.get("max_stocks", 48)),
        min_history_days=int(dcfg.get("min_history_days", 750)),
        log=log,
    )

    sentiment = None
    if bool(ncfg.get("enabled", False)) and fetch_news:
        try:
            news = fetch_and_cache_news(symbols, cache_dir_abs, int(ncfg.get("cache_days", 1)))
            sentiment = daily_sentiment_array(news, symbols, dates, float(ncfg.get("half_life_days", 2.0)))
            cov = float((np.abs(sentiment[-5:]) > 0).any(axis=0).mean())
            log(f"news sentiment ready (recent coverage {cov:.0%} of stocks)")
        except Exception as e:
            log(f"news fetching failed ({e}); continuing with zero sentiment")

    return dates, symbols, arrays, sentiment


def build_feature_set(dates, symbols, arrays, sentiment, cfg: Config):
    return build_features(dates, symbols, arrays, cfg.section("features"))


def nan_guard(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for k, v in arrays.items():
        arrays[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arrays
