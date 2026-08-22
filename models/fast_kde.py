#!/usr/bin/env python3
"""
Fast Kernel Density Estimation — Optimized for Large-Scale Sports Data.

Research basis:
- Wagner et al. (2023): Locality Sensitive Quantization for fast KDE (ICML 2023)
- Langrené & Warin (2020): Fast multivariate ECDF with KDE connection (CSDA 2021)
- Binning + FFT convolution: O(N log N) instead of O(N*M) per evaluation

Key optimization: Bin the data onto a regular grid, then use FFT-based
Gaussian convolution to evaluate the KDE at all grid points simultaneously.

This reduces per-evaluation complexity from O(N) to O(K log K) where K is
the grid size (typically 256-1024), regardless of N.

For N=100,000 data points and K=512 grid points:
- Standard KDE: 100,000 evaluations per query point
- Fast KDE: 512 * log2(512) ≈ 4,608 operations per query point
- Speedup: ~22x

Usage:
    from models.fast_kde import FastKDE
    kde = FastKDE(grid_size=512)
    kde.fit(data)
    probs = kde.evaluate(query_points)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


class FastKDE:
    """Fast 1D Kernel Density Estimation using binning + FFT convolution.

    Algorithm:
    1. Bin data onto a regular grid (histogram)
    2. Convolve with Gaussian kernel using FFT
    3. Evaluate at query points via interpolation

    Complexity: O(N + K log K) per fit, O(K) per evaluation
    where N = number of data points, K = grid size
    """

    def __init__(self, grid_size: int = 512, bandwidth: Optional[float] = None):
        """Initialize fast KDE.

        Args:
            grid_size: Number of grid points (higher = more accurate, slower)
            bandwidth: Kernel bandwidth. If None, uses Silverman's rule.
        """
        self.grid_size = grid_size
        self.bandwidth = bandwidth
        self.grid: Optional[np.ndarray] = None
        self.density: Optional[np.ndarray] = None
        self.x_min: float = 0.0
        self.x_max: float = 10.0
        self.grid_spacing: float = 0.0
        self._fitted = False

    def fit(self, data: np.ndarray, weights: Optional[np.ndarray] = None) -> "FastKDE":
        """Fit the KDE to data using binning + FFT convolution.

        Args:
            data: 1D array of data points
            weights: Optional weights for each data point

        Returns:
            self (for chaining)
        """
        data = np.asarray(data, dtype=np.float64)
        data = data[np.isfinite(data)]

        if len(data) < 2:
            # Not enough data — use uniform
            self.x_min = float(data[0]) - 1.0 if len(data) > 0 else 0.0
            self.x_max = float(data[0]) + 1.0 if len(data) > 0 else 10.0
            self.grid_spacing = (self.x_max - self.x_min) / max(self.grid_size - 1, 1)
            self.grid = np.linspace(self.x_min, self.x_max, self.grid_size)
            self.density = np.ones(self.grid_size) / self.grid_size
            self._fitted = True
            return self

        # Determine bandwidth using Silverman's rule if not provided
        if self.bandwidth is None:
            std = np.std(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            h = 0.9 * min(std, iqr / 1.34) * len(data) ** (-0.2)
            h = max(h, 0.01)  # minimum bandwidth
        else:
            h = self.bandwidth

        # Create grid with padding (3 bandwidths on each side)
        self.x_min = float(np.min(data)) - 3 * h
        self.x_max = float(np.max(data)) + 3 * h
        self.grid_spacing = (self.x_max - self.x_min) / max(self.grid_size - 1, 1)
        self.grid = np.linspace(self.x_min, self.x_max, self.grid_size)

        # Bin data onto grid (fast histogram)
        bin_indices = np.clip(
            ((data - self.x_min) / self.grid_spacing).astype(int),
            0, self.grid_size - 1
        )

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            hist = np.bincount(bin_indices, weights=weights, minlength=self.grid_size)
        else:
            hist = np.bincount(bin_indices, minlength=self.grid_size).astype(np.float64)

        # Normalize histogram
        hist = hist / (len(data) * self.grid_spacing)

        # Convolve with Gaussian kernel using FFT (this is the fast part)
        # The Gaussian kernel in frequency domain is also Gaussian
        kernel_std = h / self.grid_spacing  # in grid units

        # Use scipy's gaussian_filter1d (optimized, uses FFT internally)
        self.density = gaussian_filter1d(hist, sigma=kernel_std, mode='constant')

        # Ensure non-negative
        self.density = np.maximum(self.density, 1e-10)

        self._fitted = True
        return self

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        """Evaluate the KDE at query points using linear interpolation.

        Args:
            query_points: 1D array of points to evaluate

        Returns:
            Density values at query points
        """
        if not self._fitted:
            raise RuntimeError("KDE not fitted. Call fit() first.")

        query_points = np.asarray(query_points, dtype=np.float64)

        # Map query points to grid indices
        indices = (query_points - self.x_min) / self.grid_spacing

        # Linear interpolation
        left_idx = np.clip(np.floor(indices).astype(int), 0, self.grid_size - 2)
        right_idx = left_idx + 1
        frac = indices - left_idx

        result = self.density[left_idx] * (1 - frac) + self.density[right_idx] * frac

        return np.maximum(result, 1e-10)

    def cdf(self, query_points: np.ndarray) -> np.ndarray:
        """Evaluate the CDF at query points using cumulative trapezoidal integration.

        Args:
            query_points: 1D array of points

        Returns:
            CDF values at query points
        """
        if not self._fitted:
            raise RuntimeError("KDE not fitted. Call fit() first.")

        query_points = np.asarray(query_points, dtype=np.float64)

        # Compute CDF on grid
        cdf_grid = np.cumsum(self.density) * self.grid_spacing
        cdf_grid = np.clip(cdf_grid / max(cdf_grid[-1], 1e-10), 0, 1)

        # Interpolate to query points
        indices = (query_points - self.x_min) / self.grid_spacing
        left_idx = np.clip(np.floor(indices).astype(int), 0, self.grid_size - 2)
        right_idx = left_idx + 1
        frac = indices - left_idx

        result = cdf_grid[left_idx] * (1 - frac) + cdf_grid[right_idx] * frac

        return np.clip(result, 0, 1)

    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample from the KDE distribution.

        Uses inverse CDF method with the precomputed CDF on the grid.
        """
        if not self._fitted:
            raise RuntimeError("KDE not fitted. Call fit() first.")

        if rng is None:
            rng = np.random.default_rng()

        # Compute CDF on grid
        cdf_grid = np.cumsum(self.density) * self.grid_spacing
        cdf_grid = np.clip(cdf_grid / max(cdf_grid[-1], 1e-10), 0, 1)

        # Inverse CDF sampling
        uniform_samples = rng.uniform(0, 1, n_samples)
        indices = np.searchsorted(cdf_grid, uniform_samples)
        indices = np.clip(indices, 0, self.grid_size - 1)

        return self.grid[indices]

    def pmf_at_integers(self, max_goals: int = 10) -> np.ndarray:
        """Get probability mass at integer values (for goal distributions).

        This is the key method for football goal distributions — converts
        the continuous KDE to discrete probabilities at integer values.
        """
        integers = np.arange(0, max_goals + 1, dtype=np.float64)
        density = self.evaluate(integers)

        # Approximate PMF by integrating density around each integer
        # Using trapezoidal rule with half-integer boundaries
        boundaries = np.arange(-0.5, max_goals + 1.5, dtype=np.float64)
        cdf_vals = self.cdf(boundaries)
        pmf = np.diff(cdf_vals)

        # Normalize
        pmf = np.maximum(pmf, 1e-10)
        pmf = pmf / pmf.sum()

        return pmf


class FastKDEGoalDistribution:
    """Fast KDE for goal distributions in football matches.

    Wraps FastKDE with football-specific functionality:
    - Per-team home/away goal distributions
    - League-wide fallback
    - Score grid probability calculation
    """

    def __init__(self, grid_size: int = 256):
        self.grid_size = grid_size
        self.team_kde_home = {}
        self.team_kde_away = {}
        self.league_kde_home = None
        self.league_kde_away = None

    def fit(self, team: str, home_goals: List[float], away_goals: List[float],
            min_samples: int = 5):
        """Fit KDE for a team's goal distributions."""
        if len(home_goals) >= min_samples:
            kde = FastKDE(grid_size=self.grid_size)
            kde.fit(np.array(home_goals))
            self.team_kde_home[team] = kde

        if len(away_goals) >= min_samples:
            kde = FastKDE(grid_size=self.grid_size)
            kde.fit(np.array(away_goals))
            self.team_kde_away[team] = kde

    def fit_league(self, all_home_goals: List[float], all_away_goals: List[float]):
        """Fit league-wide KDE as fallback."""
        if len(all_home_goals) >= 10:
            self.league_kde_home = FastKDE(grid_size=self.grid_size)
            self.league_kde_home.fit(np.array(all_home_goals))

        if len(all_away_goals) >= 10:
            self.league_kde_away = FastKDE(grid_size=self.grid_size)
            self.league_kde_away.fit(np.array(all_away_goals))

    def get_pmf(self, team: str, is_home: bool, max_goals: int = 10) -> np.ndarray:
        """Get PMF for a team's goal distribution."""
        if is_home:
            kde = self.team_kde_home.get(team, self.league_kde_home)
        else:
            kde = self.team_kde_away.get(team, self.league_kde_away)

        if kde is None:
            # Fallback to Poisson(1.5)
            from scipy.stats import poisson
            return np.array([poisson.pmf(i, 1.5) for i in range(max_goals + 1)])

        return kde.pmf_at_integers(max_goals)

    def score_grid_probs(self, home_team: str, away_team: str,
                         max_goals: int = 8) -> Optional[Tuple[float, float, float]]:
        """Compute outcome probabilities using fast KDE score grid."""
        home_pmf = self.get_pmf(home_team, is_home=True, max_goals=max_goals)
        away_pmf = self.get_pmf(away_team, is_home=False, max_goals=max_goals)

        # Vectorized score grid
        outer = np.outer(home_pmf, away_pmf)

        p_home = float(np.sum(np.triu(outer, k=1)))
        p_away = float(np.sum(np.tril(outer, k=-1)))
        p_draw = float(np.sum(np.diag(outer)))

        total = p_home + p_draw + p_away
        if total > 0:
            return p_home / total, p_draw / total, p_away / total
        return None


def benchmark_kde():
    """Compare standard vs fast KDE speed."""
    import time
    from scipy.stats import gaussian_kde

    np.random.seed(42)
    sizes = [1000, 5000, 10000, 50000, 100000]

    print("KDE Benchmark: Standard vs Fast")
    print("=" * 60)
    print(f"{'N':>8} | {'Standard (ms)':>15} | {'Fast (ms)':>15} | {'Speedup':>8}")
    print("-" * 60)

    for n in sizes:
        data = np.random.gamma(2, 0.8, n)
        query = np.linspace(0, 8, 100)

        # Standard KDE
        t0 = time.perf_counter()
        try:
            kde_std = gaussian_kde(data, bw_method='silverman')
            _ = kde_std.evaluate(query)
            t_std = (time.perf_counter() - t0) * 1000
        except Exception:
            t_std = float('inf')

        # Fast KDE
        t0 = time.perf_counter()
        kde_fast = FastKDE(grid_size=512)
        kde_fast.fit(data)
        _ = kde_fast.evaluate(query)
        t_fast = (time.perf_counter() - t0) * 1000

        speedup = t_std / t_fast if t_fast > 0 else float('inf')
        print(f"{n:>8} | {t_std:>12.2f}ms | {t_fast:>12.2f}ms | {speedup:>7.1f}x")

    print("=" * 60)

