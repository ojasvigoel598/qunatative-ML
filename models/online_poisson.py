#!/usr/bin/env python3
"""
Online Poisson Ratings — Attack/Defense Strength Updated Every Match.

Research basis:
- Luis-ntonio/Match_Prediction: Online Poisson ratings improved log-loss 0.810 → 0.808
- Dixon-Coles (1997): Poisson model for football scores
- Key insight: Online ratings react to recent results instead of being frozen
  to the previous season, measurably improving probability calibration

Algorithm:
1. Each team has attack (α) and defense (δ) ratings
2. After each match, update ratings via one SGD step on goals likelihood
3. Learning rate decays with sample size (more data → smaller updates)

This is complementary to Bayesian shrinkage — the online component captures
recent form changes that Bayesian priors (which are more conservative) miss.

Usage:
    from models.online_poisson import OnlinePoissonRatings
    ratings = OnlinePoissonRatings()
    ratings.update(home_team, away_team, home_goals, away_goals, is_home=True)
    lambda_home = ratings.get_expected_goals(home_team, away_team, is_home=True)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


class OnlinePoissonRatings:
    """Online Poisson attack/defense ratings with SGD updates.

    Each team has:
    - attack rating (α): goals scored ability
    - defense rating (δ): goals conceded ability

    Expected goals:
    λ_home = exp(α_home - δ_away + home_advantage)
    λ_away = exp(α_away - δ_home)

    Updated via SGD on Poisson log-likelihood after each match.
    """

    def __init__(self, learning_rate: float = 0.1, decay: float = 0.995,
                 home_advantage: float = 0.25):
        """Initialize online Poisson ratings.

        Args:
            learning_rate: Initial learning rate for SGD updates
            decay: Learning rate decay per update
            home_advantage: Home advantage parameter (log-scale)
        """
        self.lr = learning_rate
        self.decay = decay
        self.home_adv = home_advantage
        self.ratings: Dict[str, Dict[str, float]] = {}
        self.n_updates: Dict[str, int] = {}

    def _get_or_init(self, team: str) -> Dict[str, float]:
        """Get or initialize team ratings."""
        if team not in self.ratings:
            self.ratings[team] = {"attack": 0.0, "defense": 0.0}
            self.n_updates[team] = 0
        return self.ratings[team]

    def update(self, home_team: str, away_team: str,
               home_goals: float, away_goals: float):
        """Update ratings after a match.

        Uses SGD on Poisson log-likelihood:
        L = home_goals * log(λ_home) - λ_home + away_goals * log(λ_away) - λ_away

        Gradients:
        ∂L/∂α_home = home_goals - λ_home
        ∂L/∂δ_away = -(home_goals - λ_home)
        ∂L/∂α_away = away_goals - λ_away
        ∂L/∂δ_home = -(away_goals - λ_away)
        """
        h = self._get_or_init(home_team)
        a = self._get_or_init(away_team)

        # Current expected goals
        log_lambda_home = h["attack"] - a["defense"] + self.home_adv
        log_lambda_away = a["attack"] - h["defense"]
        lambda_home = np.exp(np.clip(log_lambda_home, -10, 10))
        lambda_away = np.exp(np.clip(log_lambda_away, -10, 10))

        # Learning rate decays with number of updates
        n_h = self.n_updates[home_team]
        n_a = self.n_updates[away_team]
        lr_h = self.lr * (self.decay ** n_h)
        lr_a = self.lr * (self.decay ** n_a)

        # SGD updates
        # Home attack: ∂L/∂α_home = home_goals - λ_home
        h["attack"] += lr_h * (home_goals - lambda_home)
        # Home defense: ∂L/∂δ_home = -(away_goals - λ_away)
        h["defense"] -= lr_h * (away_goals - lambda_away)

        # Away attack: ∂L/∂α_away = away_goals - λ_away
        a["attack"] += lr_a * (away_goals - lambda_away)
        # Away defense: ∂L/∂δ_away = -(home_goals - λ_home)
        a["defense"] -= lr_a * (home_goals - lambda_home)

        # Regularize toward zero (prevent drift)
        reg = 0.001
        h["attack"] *= (1 - reg)
        h["defense"] *= (1 - reg)
        a["attack"] *= (1 - reg)
        a["defense"] *= (1 - reg)

        self.n_updates[home_team] += 1
        self.n_updates[away_team] += 1

    def get_expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Get expected goals for a match.

        Returns:
            (lambda_home, lambda_away) expected goals
        """
        h = self._get_or_init(home_team)
        a = self._get_or_init(away_team)

        log_lambda_home = h["attack"] - a["defense"] + self.home_adv
        log_lambda_away = a["attack"] - h["defense"]

        lambda_home = float(np.exp(np.clip(log_lambda_home, -10, 10)))
        lambda_away = float(np.exp(np.clip(log_lambda_away, -10, 10)))

        return max(lambda_home, 0.1), max(lambda_away, 0.1)

    def get_ratings(self, team: str) -> Dict[str, float]:
        """Get current ratings for a team."""
        h = self._get_or_init(team)
        return {
            "attack": h["attack"],
            "defense": h["defense"],
            "strength": h["attack"] - h["defense"],
            "n_updates": self.n_updates.get(team, 0),
        }

    def poisson_score_grid(self, home_team: str, away_team: str,
                           max_goals: int = 8) -> Tuple[float, float, float]:
        """Compute outcome probabilities using Poisson score grid."""
        lambda_home, lambda_away = self.get_expected_goals(home_team, away_team)

        # Vectorized Poisson PMF
        goals = np.arange(0, max_goals + 1)
        home_pmf = np.exp(-lambda_home + goals * np.log(max(lambda_home, 1e-10)) -
                         np.array([math.lgamma(g + 1) for g in goals]))
        away_pmf = np.exp(-lambda_away + goals * np.log(max(lambda_away, 1e-10)) -
                         np.array([math.lgamma(g + 1) for g in goals]))

        # Normalize
        home_pmf = home_pmf / home_pmf.sum()
        away_pmf = away_pmf / away_pmf.sum()

        # Score grid
        outer = np.outer(home_pmf, away_pmf)
        p_home = float(np.sum(np.triu(outer, k=1)))
        p_away = float(np.sum(np.tril(outer, k=-1)))
        p_draw = float(np.sum(np.diag(outer)))

        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total


# ======================================================================
# Benchmarks
# ======================================================================

def benchmark_online_poisson():
    """Benchmark online Poisson ratings."""
    import time

    ratings = OnlinePoissonRatings()

    # Simulate 1000 matches
    np.random.seed(42)
    teams = [f"Team_{i}" for i in range(20)]

    t0 = time.perf_counter()
    for _ in range(1000):
        h, a = np.random.choice(teams, 2, replace=False)
        hg = np.random.poisson(1.5)
        ag = np.random.poisson(1.2)
        ratings.update(h, a, hg, ag)
    t_update = (time.perf_counter() - t0) * 1000

    # Make predictions
    t0 = time.perf_counter()
    for _ in range(1000):
        h, a = np.random.choice(teams, 2, replace=False)
        p_h, p_d, p_a = ratings.poisson_score_grid(h, a)
    t_predict = (time.perf_counter() - t0) * 1000

    print(f"Online Poisson Ratings Benchmark:")
    print(f"  1000 updates: {t_update:.2f}ms")
    print(f"  1000 predictions: {t_predict:.2f}ms")
    print(f"  Total: {t_update + t_predict:.2f}ms")


if __name__ == "__main__":
    benchmark_online_poisson()
