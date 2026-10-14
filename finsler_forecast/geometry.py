from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import iqr
from sklearn.covariance import ledoit_wolf


def estimate_delay_ami(series: np.ndarray, max_lag: int = 15) -> int:
    """First local minimum of the average mutual information; falls back to 1."""
    n = len(series)
    iqr_val = max(float(iqr(series)), 1e-12)
    bin_width = 2.0 * iqr_val * (n ** (-1.0 / 3.0))
    span = float(series.max() - series.min())
    n_bins = int(np.clip(np.ceil(span / bin_width), 16, 48))
    bins = np.linspace(series.min(), series.max(), n_bins + 1)
    digitized = np.clip(np.digitize(series, bins) - 1, 0, n_bins - 1)

    ami = []
    for lag in range(1, min(max_lag, n // 10) + 1):
        x1, x2 = digitized[:-lag], digitized[lag:]
        n_p = len(x1)
        p_joint = np.zeros((n_bins, n_bins))
        np.add.at(p_joint, (x1, x2), 1.0)
        p_joint /= n_p
        p_x1 = np.bincount(x1, minlength=n_bins) / n_p
        p_x2 = np.bincount(x2, minlength=n_bins) / n_p
        p_outer = np.outer(p_x1, p_x2)
        nz = p_joint > 0
        ami.append(float(np.sum(p_joint[nz] * np.log2(p_joint[nz] / (p_outer[nz] + 1e-15)))))

    derivs = np.diff(ami)
    if len(derivs) < 2:
        return 1
    local_min = np.where((derivs[:-1] < 0) & (derivs[1:] > 0))[0] + 1
    return int(local_min[0] + 1) if len(local_min) > 0 else 1


def finsler_f_and_grad(delta: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Randers norm F(y) = sqrt(y^T a y) + b.y for each row y of delta, and its gradient."""
    alpha = np.sqrt(np.einsum("ik,kl,il->i", delta, a, delta) + 1e-18)
    f = alpha + delta @ b
    grad = (delta @ a) / alpha[:, None] + b
    return f, grad


@dataclass
class LocalGeometry:
    a: np.ndarray       # local Riemannian metric
    a_inv: np.ndarray   # local covariance (a^-1)
    b: np.ndarray       # Randers covector, b = -wind
    strength: float     # ||b||_a, in [0, 0.95]
    drift: np.ndarray   # mean forward state displacement of the neighbours
    next_ret_t: float   # t-stat of the neighbours' next-day return


class FinslerRandersGeodesicEngine:
    """
    Delay-embeds log-returns in R^3, fits a local Randers (Finsler) metric from the k
    nearest neighbours' covariance and drift ("wind"), and forecasts by following the
    true historical continuations of the closest analog states. No synthetic sampling,
    no training.
    """

    def __init__(
        self,
        ticker: str,
        asset_name: str,
        years: int = 5,
        k_neighbors: int = 30,
        n_scenarios: int = 7,
        tau: int = 1,
        freq: str = "B",
        drop_last_bar: bool = True,
    ) -> None:
        self.ticker = ticker
        self.asset_name = asset_name
        self.years = int(np.clip(years, 1, 20))
        self.k_neighbors = int(np.clip(k_neighbors, 10, 200))
        self.n_scenarios = int(np.clip(n_scenarios, 3, 20))
        self.tau_request = int(tau)  # <= 0 means "estimate with AMI"
        self.freq = "D" if freq == "D" else "B"
        self.drop_last_bar = drop_last_bar

        self.dates = pd.DatetimeIndex([])
        self.prices = np.array([])
        self.log_returns = np.array([])
        self.scale = 1.0

        self.tau = 1
        self.m = 3
        self.point_cloud = np.empty((0, 0))
        self.tangent_velocities = np.empty((0, 0))  # forward differences x[j+1] - x[j]
        self.tree: cKDTree | None = None

        self.latest_geometry: LocalGeometry | None = None
        self.latest_wind_covector = np.zeros(3)
        self.wind_magnitude = 0.0
        self.wind_direction_bias = "neutral"  # "bullish" | "bearish" | "neutral"

        self.k_used = 0
        self.analog_indices = np.array([], dtype=int)
        self.forecast_dates = pd.DatetimeIndex([])
        self.scenario_states = np.empty((0, 0, 0))  # (K, steps+1, m)
        self.scenario_prices = np.empty((0, 0))
        self.frechet_mean_price = np.array([])
        self.karcher_mean_price = np.array([])
        self.manifold_median_price = np.array([])

    def fetch_and_reconstruct(self, price_df: pd.DataFrame) -> "FinslerRandersGeodesicEngine":
        df = price_df[["Close"]].dropna()
        df = df[df["Close"] > 0]

        # Yahoo's last daily bar can be an unfinished intraday bar
        if self.drop_last_bar and len(df) > 0 and df.index[-1].normalize() >= pd.Timestamp.today().normalize():
            df = df.iloc[:-1]

        if len(df) < 150:
            raise ValueError(f"insufficient history for {self.asset_name} ({len(df)} bars, need >= 150)")

        self.dates = pd.DatetimeIndex(df.index)
        self.prices = df["Close"].to_numpy(dtype=np.float64)
        self.log_returns = np.diff(np.log(self.prices))
        self.scale = max(float(np.std(self.log_returns)), 1e-12)
        self.tau = (
            estimate_delay_ami(self.log_returns)
            if self.tau_request <= 0
            else int(np.clip(self.tau_request, 1, 30))
        )

        n = len(self.log_returns)
        n_pts = n - (self.m - 1) * self.tau
        if n_pts < 5 * self.k_neighbors:
            raise ValueError("not enough embedded points for the chosen tau / k_neighbors")

        # x_j = [r(t), r(t - tau), r(t - 2*tau)], column 0 = newest return
        self.point_cloud = np.column_stack(
            [
                self.log_returns[(self.m - 1 - i) * self.tau : (self.m - 1 - i) * self.tau + n_pts]
                for i in range(self.m)
            ]
        )
        self.tangent_velocities = np.diff(self.point_cloud, axis=0)
        self.tree = cKDTree(self.point_cloud)
        return self

    def _local_geometry(self, pt: np.ndarray) -> LocalGeometry:
        n_pts = len(self.point_cloud)
        k = self.k_neighbors
        _, idx = self.tree.query(pt, k=min(n_pts, k + 1))
        idx = np.atleast_1d(idx)
        idx = idx[idx <= n_pts - 2][:k]  # a forward step must exist
        if len(idx) < 8:
            raise ValueError("too few neighbours with a forward step")

        pts = self.point_cloud[idx]
        cov, _ = ledoit_wolf(pts)
        if not np.isfinite(cov).all() or np.trace(cov) <= 0:
            raise ValueError("degenerate local neighbourhood (zero variance)")
        a = np.linalg.pinv(cov)
        a = 0.5 * (a + a.T)

        vels = self.tangent_velocities[idx]
        mu = vels.mean(axis=0)
        cov_v, _ = ledoit_wolf(vels)
        rho = float(np.sqrt(max(mu @ np.linalg.pinv(cov_v) @ mu, 0.0)))  # drift in units of flow dispersion
        mu_a = float(np.sqrt(max(mu @ a @ mu, 0.0)))
        if mu_a < 1e-15:
            strength, b = 0.0, np.zeros(self.m)
        else:
            strength = float(min(np.tanh(rho), 0.95))  # ||b||_a < 1, so F stays a Finsler norm
            b = -strength * (a @ mu) / mu_a  # Zermelo navigation: b = -wind

        next_ret = self.point_cloud[idx + 1, 0]
        se = float(np.std(next_ret, ddof=1)) / np.sqrt(len(next_ret)) + 1e-15
        t_stat = float(np.mean(next_ret) / se)
        return LocalGeometry(a=a, a_inv=cov, b=b, strength=strength, drift=mu, next_ret_t=t_stat)

    def _select_analogs(self, x_latest: np.ndarray, geom: LocalGeometry, horizon: int) -> np.ndarray:
        """K nearest, temporally de-correlated historical states with `horizon` future steps available."""
        n_pts = len(self.point_cloud)
        theiler = max(10, (self.m - 1) * self.tau + 1)
        last_ok = min(n_pts - 1 - horizon, n_pts - 2 - theiler)
        pool = min(n_pts, max(10 * self.n_scenarios, 100))
        _, idx = self.tree.query(x_latest, k=pool)
        idx = np.atleast_1d(idx)
        idx = idx[idx <= last_ok]
        if len(idx) == 0:
            raise ValueError("no usable analog states")

        d = self.point_cloud[idx] - x_latest
        maha = np.einsum("ik,kl,il->i", d, geom.a, d)
        order = np.argsort(maha, kind="stable")

        chosen: list[int] = []
        for j in idx[order]:
            j = int(j)
            if all(abs(j - c) > theiler for c in chosen):
                chosen.append(j)
            if len(chosen) == self.n_scenarios:
                break
        if len(chosen) < 3:
            raise ValueError("fewer than 3 de-correlated analog states found")
        return np.array(chosen, dtype=int)

    def _geometric_centers(self, pts: np.ndarray, g: LocalGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Frechet (Finsler), Karcher (Riemannian) and geometric-median centers, in the frozen chart at their mean."""
        s = self.scale
        p0 = pts.mean(axis=0)
        xu = (pts - p0) / s
        a_u = g.a * s * s
        b_u = g.b * s
        u0 = np.zeros(self.m)

        def frechet(u: np.ndarray):
            f, gf = finsler_f_and_grad(u[None, :] - xu, a_u, b_u)
            return float(np.mean(f**2)), np.mean(2.0 * f[:, None] * gf, axis=0)

        def median(u: np.ndarray):
            eps = 1e-6
            f, gf = finsler_f_and_grad(u[None, :] - xu, a_u, b_u)
            r = np.sqrt(f**2 + eps**2)
            return float(np.sum(r)), np.sum((f / r)[:, None] * gf, axis=0)

        def solve(fun) -> np.ndarray:
            f0 = fun(u0)[0]
            res = minimize(fun, u0, jac=True, method="BFGS", options={"gtol": 1e-10, "maxiter": 300})
            ok = np.all(np.isfinite(res.x)) and res.fun <= f0 + 1e-15  # never worse than the start
            return res.x if ok else u0

        u_frechet = solve(frechet)
        u_median = solve(median)

        u_karcher = u0.copy()
        for _ in range(100):
            step = (xu - u_karcher).mean(axis=0)
            u_karcher = u_karcher + step
            if float(np.sqrt(step @ a_u @ step)) < 1e-12:
                break

        return p0 + s * u_frechet, p0 + s * u_karcher, p0 + s * u_median

    def _path_to_price(self, states: np.ndarray) -> np.ndarray:
        returns = states[1:, 0]
        return self.prices[-1] * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))

    def simulate(self, horizon_days: int = 3) -> "FinslerRandersGeodesicEngine":
        h = int(horizon_days)
        x_latest = self.point_cloud[-1].copy()

        geom = self._local_geometry(x_latest)
        self.latest_geometry = geom
        self.wind_magnitude = geom.strength
        self.latest_wind_covector = -geom.b * self.scale
        if geom.next_ret_t > 2.0:
            self.wind_direction_bias = "bullish"
        elif geom.next_ret_t < -2.0:
            self.wind_direction_bias = "bearish"
        else:
            self.wind_direction_bias = "neutral"

        analog = self._select_analogs(x_latest, geom, h)
        self.analog_indices = analog
        self.k_used = len(analog)

        states = np.empty((self.k_used, h + 1, self.m))
        states[:, 0, :] = x_latest
        for step in range(1, h + 1):
            states[:, step, :] = self.point_cloud[analog + step]
        self.scenario_states = states
        self.scenario_prices = np.stack([self._path_to_price(states[k]) for k in range(self.k_used)])

        frechet = np.tile(x_latest, (h + 1, 1))
        karcher = np.tile(x_latest, (h + 1, 1))
        median = np.tile(x_latest, (h + 1, 1))
        for step in range(1, h + 1):
            pts = states[:, step, :]
            g_step = self._local_geometry(pts.mean(axis=0))
            frechet[step], karcher[step], median[step] = self._geometric_centers(pts, g_step)

        self.frechet_mean_price = self._path_to_price(frechet)
        self.karcher_mean_price = self._path_to_price(karcher)
        self.manifold_median_price = self._path_to_price(median)

        last = self.dates[-1]
        offset = pd.offsets.Day if self.freq == "D" else pd.offsets.BDay
        self.forecast_dates = pd.DatetimeIndex([last] + [last + offset(i) for i in range(1, h + 1)])
        return self
