from __future__ import annotations

import pandas as pd

from finsler_forecast.diffusion.denoise import detect_closures, flat_candle_mask, remove_flat_candles


def _bar(minute: int, price: float, volume: float, high: float | None = None, low: float | None = None):
    return {
        "Datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute),
        "Open": price,
        "High": high if high is not None else price,
        "Low": low if low is not None else price,
        "Close": price,
        "Volume": volume,
    }


def _frame(bars: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(bars)


def test_flat_candle_mask_flags_zero_volume_zero_range():
    df = _frame(
        [
            _bar(0, 1.1000, volume=10),
            _bar(1, 1.1000, volume=0),
            _bar(2, 1.1005, volume=0, high=1.1006, low=1.1004),  # zero volume but real range: not flat
        ]
    )
    mask = flat_candle_mask(df)
    assert mask.tolist() == [False, True, False]


def test_detect_closures_ignores_runs_shorter_than_threshold():
    df = _frame([_bar(0, 1.1, volume=10)] + [_bar(i, 1.1, volume=0) for i in range(1, 10)])
    closures = detect_closures(df, min_duration_minutes=30)
    assert len(closures) == 0


def test_detect_closures_finds_a_single_long_run():
    bars = [_bar(0, 1.1, volume=10)]
    bars += [_bar(i, 1.1050, volume=0) for i in range(1, 41)]  # 40-minute closure
    bars += [_bar(41, 1.1060, volume=10)]
    df = _frame(bars)

    closures = detect_closures(df, min_duration_minutes=30)

    assert len(closures) == 1
    row = closures.iloc[0]
    assert row["flat_minutes"] == 40
    assert row["price"] == 1.1050
    assert row["start_time"] == pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=1)
    assert row["end_time"] == pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=40)


def test_detect_closures_finds_multiple_separate_runs():
    bars = [_bar(0, 1.1, volume=10)]
    bars += [_bar(i, 1.1, volume=0) for i in range(1, 35)]  # closure 1: 34 min
    bars += [_bar(35, 1.105, volume=10)]
    bars += [_bar(i, 1.105, volume=0) for i in range(36, 70)]  # closure 2: 34 min
    bars += [_bar(70, 1.11, volume=10)]
    df = _frame(bars)

    closures = detect_closures(df, min_duration_minutes=30)
    assert len(closures) == 2
    assert closures["flat_minutes"].tolist() == [34, 34]


def test_detect_closures_handles_closure_at_start_of_series():
    bars = [_bar(i, 1.1, volume=0) for i in range(0, 35)] + [_bar(35, 1.105, volume=10)]
    df = _frame(bars)

    closures = detect_closures(df, min_duration_minutes=30)
    assert len(closures) == 1
    assert closures.iloc[0]["flat_minutes"] == 35


def test_remove_flat_candles_drops_every_flat_row_regardless_of_run_length():
    df = _frame(
        [
            _bar(0, 1.1, volume=10),
            _bar(1, 1.1, volume=0),  # short flat run, below any closure threshold
            _bar(2, 1.1010, volume=10),
        ]
    )
    cleaned = remove_flat_candles(df)
    assert cleaned["Volume"].tolist() == [10, 10]
    assert cleaned["Close"].tolist() == [1.1, 1.1010]


def _three_hour_bars_with_a_fully_closed_middle_hour() -> pd.DataFrame:
    bars = [_bar(m, 1.1000 + m * 1e-5, volume=10) for m in range(0, 5)]  # real trading in hour 0
    bars += [_bar(60 + m, 1.1050, volume=0) for m in range(0, 60)]  # hour 1 entirely closed
    bars += [_bar(120 + m, 1.1100 + m * 1e-5, volume=10) for m in range(0, 5)]  # real trading in hour 2
    return _frame(bars)


def test_resample_to_hourly_on_raw_data_keeps_the_fake_closed_hour():
    """Documents the raw variant's known limitation: a closed-market hour still resamples into a bar."""
    from finsler_forecast.diffusion.dataset import resample_to_hourly

    hourly = resample_to_hourly(_three_hour_bars_with_a_fully_closed_middle_hour())

    assert len(hourly) == 3
    assert pd.Timestamp("2024-01-01 01:00:00") in hourly.index


def test_resample_to_hourly_on_denoised_data_drops_the_closed_hour():
    from finsler_forecast.diffusion.dataset import resample_to_hourly

    df = _three_hour_bars_with_a_fully_closed_middle_hour()
    hourly = resample_to_hourly(remove_flat_candles(df))

    assert len(hourly) == 2  # the fully-closed hour must not appear
    assert pd.Timestamp("2024-01-01 01:00:00") not in hourly.index


def test_build_denoised_dataset_writes_a_file_with_flat_rows_removed(tmp_path):
    from finsler_forecast.diffusion.dataset import build_denoised_dataset

    raw_path = tmp_path / "raw.parquet"
    out_path = tmp_path / "denoised.parquet"
    df = _three_hour_bars_with_a_fully_closed_middle_hour()
    df.to_parquet(raw_path, index=False)

    build_denoised_dataset(raw_parquet=str(raw_path), out_path=str(out_path))

    result = pd.read_parquet(out_path)
    assert len(result) == len(df) - 60
    assert (result["Volume"] > 0).all()
