"""
AI complement on top of the Finsler/analog-flow geometry — not a replacement.

Why this exists
----------------
The honest walk-forward numbers in `benchmark.py` show the Finsler engine's own
wind/Randers term (frechet, median) consistently *hurts* vs. naive persistence,
while the plain geometric mean of the analog neighbours (karcher) is the one
component that is occasionally competitive. So the baseline worth protecting is
karcher, not the full Finsler forecast.

This module adds a small, simple (shallow gradient-boosted trees, not a deep
net) residual model on top of karcher: it is trained to predict the error
karcher is about to make, using features already sitting inside the engine
(wind strength, neighbour spread, local-metric anisotropy) plus a few plain
technical features. It is combined with karcher through a blend weight chosen
by nested validation — i.e. the weight that would have worked on *past* folds,
never the weight that works best on the fold being scored. If no weight beats
karcher alone on the validation folds, the weight is forced to 0 (pure
karcher): the ensemble is explicitly not allowed to be sold as better than the
baseline unless it demonstrably was, out of sample, before being tested.

No econometric volatility models (GARCH/EGARCH/...) are used anywhere here —
only engine-internal geometry features, plain price-derived technicals, and a
tree ensemble, per the brief that the complement should be AI-side, not
econometrics-side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from geometry import FinslerRandersGeodesicEngine


def engine_meta_features(engine: FinslerRandersGeodesicEngine) -> dict[str, float]:
    """Cheap, already-computed-by-the-engine features describing *how much to trust*
    the current analog-flow forecast, not a new price model."""
    geom = engine.latest_geometry
    scenario_rets_h = np.log(engine.scenario_prices[:, -1] / engine.prices[-1])
    eigvals = np.linalg.eigvalsh(geom.a_inv) if geom is not None else np.array([1.0, 1.0, 1.0])
    eigvals = np.clip(eigvals, 1e-12, None)
    return {
        "wind_strength": float(geom.strength) if geom is not None else 0.0,
        "wind_t_stat": float(np.clip(geom.next_ret_t, -10, 10)) if geom is not None else 0.0,
        "neighbour_anisotropy": float(eigvals.max() / eigvals.min()),  # 1 = isotropic, large = stretched metric
        "analog_spread": float(np.std(scenario_rets_h)),
        "analog_mean_ret": float(np.mean(scenario_rets_h)),
        "k_used": float(engine.k_used),
        "recent_vol_5": float(np.std(engine.log_returns[-5:])) if len(engine.log_returns) >= 5 else 0.0,
        "recent_vol_21": float(np.std(engine.log_returns[-21:])) if len(engine.log_returns) >= 21 else 0.0,
        "momentum_5": float(np.sum(engine.log_returns[-5:])) if len(engine.log_returns) >= 5 else 0.0,
        "momentum_21": float(np.sum(engine.log_returns[-21:])) if len(engine.log_returns) >= 21 else 0.0,
    }


FEATURE_NAMES = [
    "wind_strength", "wind_t_stat", "neighbour_anisotropy", "analog_spread", "analog_mean_ret",
    "k_used", "recent_vol_5", "recent_vol_21", "momentum_5", "momentum_21",
]


@dataclass
class EnsembleResult:
    karcher_pred: np.ndarray
    ai_blend_pred: np.ndarray
    blend_weight: float


def _residual_model() -> HistGradientBoostingRegressor:
    # Deliberately shallow/small: the literature point that motivated this (simpler models
    # generalize better on noisy financial residuals) argues against a deep or wide model here.
    return HistGradientBoostingRegressor(max_depth=3, max_iter=60, learning_rate=0.08, min_samples_leaf=15)


def _direction_flip_model() -> HistGradientBoostingClassifier:
    """Classifies whether karcher's directional call was wrong, from the same meta-features.
    Used only to flip the *direction* label when confident, never the predicted price level —
    so MAE is identical to karcher's by construction and can't be made worse by this model."""
    return HistGradientBoostingClassifier(max_depth=3, max_iter=60, learning_rate=0.08, min_samples_leaf=15)


def pick_blend_weight(
    karcher_err: np.ndarray,
    ai_err: np.ndarray,
    weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    min_relative_gain: float = 0.05,
) -> float:
    """Weight w minimizing MAE of (1-w)*karcher_err + w*ai_err, restricted to candidates that beat
    w=0 (pure karcher) by at least `min_relative_gain` on held-out validation folds. A margin, not
    just "any improvement", because a tiny validation-set edge is noise and shouldn't be trusted
    to survive onto the next, still-unseen fold — in practice a razor-thin edge rarely did here."""
    base_mae = float(np.mean(np.abs(karcher_err)))
    best_w, best_mae = 0.0, base_mae
    for w in weights:
        blended = (1 - w) * karcher_err + w * ai_err
        mae = float(np.mean(np.abs(blended)))
        if mae < best_mae * (1 - min_relative_gain):
            best_w, best_mae = w, mae
    return best_w
