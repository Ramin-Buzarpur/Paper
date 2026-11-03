import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble import FEATURE_NAMES, engine_meta_features, pick_blend_weight  # noqa: E402
from geometry import FinslerRandersGeodesicEngine  # noqa: E402


def _synthetic_price_df(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame({"Close": prices}, index=dates)


def test_engine_meta_features_has_all_declared_names():
    engine = (
        FinslerRandersGeodesicEngine(ticker="SYN", asset_name="Synthetic", k_neighbors=15, n_scenarios=5)
        .fetch_and_reconstruct(_synthetic_price_df())
        .simulate(3)
    )
    feats = engine_meta_features(engine)
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in feats.values())


def test_pick_blend_weight_requires_a_margin_not_just_any_improvement():
    rng = np.random.default_rng(0)
    karcher_err = rng.normal(0, 1.0, 200)
    # ai_err barely better (1% smaller spread) -- should NOT clear the 5% margin
    ai_err_barely_better = karcher_err * 0.99
    assert pick_blend_weight(karcher_err, ai_err_barely_better) == 0.0

    # ai_err meaningfully better (50% smaller) -- should clear the margin and be picked
    ai_err_much_better = karcher_err * 0.5
    assert pick_blend_weight(karcher_err, ai_err_much_better) > 0.0


def test_pick_blend_weight_never_picks_a_worse_candidate():
    rng = np.random.default_rng(1)
    karcher_err = rng.normal(0, 1.0, 200)
    ai_err_worse = karcher_err * 2.0
    assert pick_blend_weight(karcher_err, ai_err_worse) == 0.0
