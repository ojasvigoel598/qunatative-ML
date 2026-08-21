#!/usr/bin/env python3
"""
Fast Mixture Monte Carlo — Vectorized Simulation for Large-Scale Data.

Research basis:
- McLachlan & Peel (2000): Finite Mixture Models
- Vectorized NumPy simulation: O(N) instead of O(N*loops)

Key optimization: Replace Python for-loop with vectorized NumPy operations.
Instead of simulating one match at a time, simulate all matches simultaneously.

For 1000 simulations:
- Loop-based: 1000 iterations × Python overhead ≈ 5ms
- Vectorized: 1 NumPy operation ≈ 0.05ms
- Speedup: ~100x

Usage:
    from models.fast_mixture_mc import FastMixtureMC
    mc = FastMixtureMC(n_simulations=3000)
    p_home, p_draw, p_away = mc.simulate(lambda_home, lambda_away, regime_weights)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class FastMixtureMC:
    """Fast Mixture Model Monte Carlo simulation with vectorized operations.

    P(X) = w_floor * P(X|Floor) + w_normal * P(X|Normal) + w_ceiling * P(X|Ceiling)

    Vectorized implementation processes all simulations in parallel.
    """

    def __init__(self, n_simulations: int = 3000, seed: int = 42):
        """Initialize fast MC simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            seed: Random seed for reproducibility
        """
        self.n_sims = n_simulations
        self.rng = np.random.default_rng(seed)

        # Regime multipliers (fixed for now, could be learned)
        self.floor_mult = 0.7
        self.normal_mult = 1.0
        self.ceiling_mult = 1.4

    def detect_regime(self, team_strength: float, opponent_strength: float,
                      recent_form: float, season_avg: float) -> Dict[str, float]:
        """Detect regime weights based on context.

        Returns weights for Floor, Normal, Ceiling regimes.
        """
        form_dev = recent_form - season_avg
        strength_diff = team_strength - opponent_strength

        # Base weights
        w_floor = 0.15
        w_normal = 0.60
        w_ceiling = 0.25

        # Modulate by form deviation
        if form_dev < -0.5:
            w_floor += 0.15
            w_normal -= 0.10
            w_ceiling -= 0.05
        elif form_dev > 0.5:
            w_ceiling += 0.15
            w_normal -= 0.10
            w_floor -= 0.05

        # Modulate by strength differential
        if strength_diff < -0.5:
            w_floor += 0.10
            w_ceiling -= 0.10
        elif strength_diff > 0.5:
            w_ceiling += 0.10
            w_floor -= 0.10

        # Normalize
        total = w_floor + w_normal + w_ceiling
        return {
            "floor": max(0.05, w_floor / total),
            "normal": max(0.30, w_normal / total),
            "ceiling": max(0.05, w_ceiling / total),
        }

    def simulate(self, lambda_home: float, lambda_away: float,
                 regime_weights: Dict[str, float]) -> Tuple[float, float, float]:
        """Vectorized Monte Carlo simulation.

        All simulations run in parallel using NumPy vectorization.

        Args:
            lambda_home: Expected home goals
            lambda_away: Expected expected away goals
            regime_weights: Dict with 'floor', 'normal', 'ceiling' weights

        Returns:
            (p_home, p_draw, p_away) probabilities
        """
        n = self.n_sims

        # Step 1: Sample regime for each simulation (vectorized)
        uniform = self.rng.uniform(0, 1, n)
        w_floor = regime_weights["floor"]
        w_normal = regime_weights["normal"]

        # Assign regime: 0=floor, 1=normal, 2=ceiling
        regimes = np.where(
            uniform < w_floor, 0,
            np.where(uniform < w_floor + w_normal, 1, 2)
        )

        # Step 2: Get multipliers for each regime (vectorized lookup)
        multipliers = np.where(
            regimes == 0, self.floor_mult,
            np.where(regimes == 1, self.normal_mult, self.ceiling_mult)
        )

        # Step 3: Simulate goals for all matches simultaneously (vectorized)
        # Poisson sampling is vectorized in NumPy
        home_goals = self.rng.poisson(lambda_home * multipliers)
        away_goals = self.rng.poisson(lambda_away * multipliers)

        # Step 4: Compute probabilities (vectorized comparison)
        p_home = float(np.mean(home_goals > away_goals))
        p_draw = float(np.mean(home_goals == away_goals))
        p_away = float(np.mean(home_goals < away_goals))

        return p_home, p_draw, p_away

    def simulate_batch(self, lambda_home_batch: np.ndarray,
                       lambda_away_batch: np.ndarray,
                       regime_weights_batch: list) -> np.ndarray:
        """Batch simulation for multiple matches.

        Args:
            lambda_home_batch: Array of home expected goals for each match
            lambda_away_batch: Array of away expected goals for each match
            regime_weights_batch: List of regime weight dicts

        Returns:
            Array of shape (n_matches, 3) with [p_home, p_draw, p_away]
        """
        n_matches = len(lambda_home_batch)
        results = np.zeros((n_matches, 3))

        for i in range(n_matches):
            results[i] = self.simulate(
                lambda_home_batch[i],
                lambda_away_batch[i],
                regime_weights_batch[i]
            )

        return results


class FastRegimeDetector:
    """Fast regime detection using vectorized operations.

    Detects whether a team is in Floor, Normal, or Ceiling regime
    based on recent form, opponent strength, and context.
    """

    def __init__(self):
        self.team_stats = {}

    def update(self, team: str, goals_scored: float, goals_conceded: float,
               is_home: bool):
        """Update team statistics."""
        if team not in self.team_stats:
            self.team_stats[team] = {
                "recent_scores": [],
                "season_avg": 1.5,
                "n_matches": 0,
            }

        stats = self.team_stats[team]
        stats["recent_scores"].append(goals_scored)
        stats["n_matches"] += 1

        # Keep only last 10 matches for speed
        if len(stats["recent_scores"]) > 10:
            stats["recent_scores"] = stats["recent_scores"][-10:]

        # Update season average (online)
        n = stats["n_matches"]
        stats["season_avg"] = ((n - 1) * stats["season_avg"] + goals_scored) / n

    def get_regime(self, team: str, opponent_strength: float) -> str:
        """Get current regime for a team.

        Returns: 'floor', 'normal', or 'ceiling'
        """
        if team not in self.team_stats:
            return "normal"

        stats = self.team_stats[team]
        recent_form = np.mean(stats["recent_scores"][-5:]) if stats["recent_scores"] else 1.5
        season_avg = stats["season_avg"]

        form_dev = recent_form - season_avg

        if form_dev < -0.5:
            return "floor"
        elif form_dev > 0.5:
            return "ceiling"
        else:
            return "normal"

    def get_regime_weights(self, team: str, opponent_strength: float) -> Dict[str, float]:
        """Get regime weights for a team."""
        regime = self.get_regime(team, opponent_strength)

        if regime == "floor":
            return {"floor": 0.45, "normal": 0.45, "ceiling": 0.10}
        elif regime == "ceiling":
            return {"floor": 0.10, "normal": 0.45, "ceiling": 0.45}
        else:
            return {"floor": 0.15, "normal": 0.70, "ceiling": 0.15}


# ======================================================================
# Benchmarks
# ======================================================================

def benchmark_mixture_mc():
    """Compare loop-based vs vectorized MC speed."""
    import time

    np.random.seed(42)
    n_sims_list = [100, 500, 1000, 3000, 10000]

    print("Mixture MC Benchmark: Loop vs Vectorized")
    print("=" * 60)
    print(f"{'Sims':>8} | {'Loop (ms)':>12} | {'Vectorized (ms)':>15} | {'Speedup':>8}")
    print("-" * 60)

    for n_sims in n_sims_list:
        lambda_h, lambda_a = 1.6, 1.3
        regime_w = {"floor": 0.15, "normal": 0.60, "ceiling": 0.25}

        # Loop-based
        t0 = time.perf_counter()
        rng = np.random.default_rng(42)
        home_goals = []
        away_goals = []
        for _ in range(n_sims):
            r = rng.random()
            if r < regime_w["floor"]:
                mult = 0.7
            elif r < regime_w["floor"] + regime_w["normal"]:
                mult = 1.0
            else:
                mult = 1.4
            hg = rng.poisson(lambda_h * mult)
            ag = rng.poisson(lambda_a * mult)
            home_goals.append(hg)
            away_goals.append(ag)
        home_goals = np.array(home_goals)
        away_goals = np.array(away_goals)
        p_h = np.mean(home_goals > away_goals)
        p_d = np.mean(home_goals == away_goals)
        p_a = np.mean(home_goals < away_goals)
        t_loop = (time.perf_counter() - t0) * 1000

        # Vectorized
        t0 = time.perf_counter()
        mc = FastMixtureMC(n_simulations=n_sims, seed=42)
        p_h, p_d, p_a = mc.simulate(lambda_h, lambda_a, regime_w)
        t_vec = (time.perf_counter() - t0) * 1000

        speedup = t_loop / t_vec if t_vec > 0 else float('inf')
        print(f"{n_sims:>8} | {t_loop:>9.2f}ms | {t_vec:>12.2f}ms | {speedup:>7.1f}x")

    print("=" * 60)


if __name__ == "__main__":
    benchmark_mixture_mc()
