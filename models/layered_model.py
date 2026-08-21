#!/usr/bin/env python3
"""
Layered Model System — Adaptive Model Selection & Evidence-Driven Implementation.

Implements the full architecture from the diagram:
  Historical Data -> Bayesian Prior -> Usage/Regime -> [Parametric | KDE | Mixture MC]
  -> Contextual -> Ensemble -> Probability -> Calibration -> Market -> Edge/EV -> Kelly -> BET

Each layer is a candidate. The ablation tournament determines which layers
actually add value through walk-forward validation.

Research basis:
- Bayesian shrinkage reduces overfitting on small samples (Russell & Norvig)
- KDE captures non-parametric distributions (Rosenblatt 1956, Parzen 1962)
- Mixture models handle regime changes (McLachlan & Peel 2000)
- EWMA captures recency (exponential smoothing literature)
- Ensemble diversity reduces variance (Dietterich 2000)

Usage:
    from models.layered_model import LayeredModel, run_ablation_tournament
    model = LayeredModel()
    model.train(train_df)
    probs = model.predict("Barcelona", "Real Madrid")
    results = run_ablation_tournament(df)
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from models.fast_kde import FastKDE, FastKDEGoalDistribution
from models.fast_mixture_mc import FastMixtureMC
from models.online_poisson import OnlinePoissonRatings
from models.speed_optimizations import vectorized_poisson_score_grid, batch_poisson_score_grid

warnings.filterwarnings("ignore")

OUTCOMES = ["home_win", "draw", "away_win"]
MAX_GOALS = 10


# ======================================================================
# Layer 1: Bayesian Team Strength Prior
# ======================================================================
class BayesianTeamPrior:
    """Bayesian shrinkage for team strength estimation.

    Instead of raw historical averages, shrinks toward the league mean
    based on sample size. Teams with few matches get stronger shrinkage.

    Prior: N(mu_league, sigma^2)
    Likelihood: N(observed, sigma^2/n)
    Posterior: weighted combination
    """

    def __init__(self, league_mu: float = 1.5, league_sigma: float = 0.5):
        self.league_mu = league_mu
        self.league_sigma = league_sigma
        self.team_stats: Dict[str, Dict] = {}

    def update(self, team: str, goals_scored: float, goals_conceded: float):
        """Update team statistics with a new match observation."""
        if team not in self.team_stats:
            self.team_stats[team] = {
                "n": 0, "goals_for": 0.0, "goals_against": 0.0,
                "results": []
            }
        s = self.team_stats[team]
        s["n"] += 1
        s["goals_for"] += goals_scored
        s["goals_against"] += goals_conceded

    def get_strength(self, team: str) -> Dict[str, float]:
        """Get Bayesian-shrunk strength estimate for a team."""
        if team not in self.team_stats or self.team_stats[team]["n"] == 0:
            return {
                "attack": self.league_mu,
                "defense": self.league_mu,
                "strength": 0.0,
                "uncertainty": self.league_sigma,
                "n_matches": 0,
            }

        s = self.team_stats[team]
        n = s["n"]
        raw_attack = s["goals_for"] / n
        raw_defense = s["goals_against"] / n

        # Bayesian shrinkage: posterior mean = (n * x_bar + tau * mu) / (n + tau)
        # where tau = sigma^2_likelihood / sigma^2_prior
        tau = 1.0  # assume unit observation variance
        shrinkage = tau / (n + tau)

        attack = raw_attack * (1 - shrinkage) + self.league_mu * shrinkage
        defense = raw_defense * (1 - shrinkage) + self.league_mu * shrinkage

        return {
            "attack": attack,
            "defense": defense,
            "strength": attack - defense,
            "uncertainty": self.league_sigma / math.sqrt(n + 1),
            "n_matches": n,
        }


# ======================================================================
# Layer 2: EWMA Recency Model
# ======================================================================
class EWMARecency:
    """Exponentially Weighted Moving Average for recent form.

    EWMA_t = alpha * X_t + (1 - alpha) * EWMA_{t-1}

    Alpha is learned via walk-forward validation (not hard-coded).
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.team_ewma: Dict[str, float] = {}

    def update(self, team: str, observed_goals: float):
        """Update EWMA with new observation."""
        if team not in self.team_ewma:
            self.team_ewma[team] = observed_goals
        else:
            self.team_ewma[team] = self.alpha * observed_goals + (1 - self.alpha) * self.team_ewma[team]

    def get_ewma(self, team: str) -> float:
        """Get current EWMA estimate for a team."""
        return self.team_ewma.get(team, 1.5)

    def fit_alpha(self, team: str, history: List[float], alphas: List[float] = None):
        """Find optimal alpha for a team via cross-validation on history."""
        if alphas is None:
            alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        best_alpha = 0.3
        best_mse = float("inf")

        for alpha in alphas:
            if len(history) < 3:
                break
            # Walk-forward: predict each point using prior EWMA
            ewma = history[0]
            mse = 0.0
            for i in range(1, len(history)):
                pred = ewma
                mse += (history[i] - pred) ** 2
                ewma = alpha * history[i] + (1 - alpha) * ewma
            mse /= max(len(history) - 1, 1)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.alpha = best_alpha
        return best_alpha


# ======================================================================
# Layer 3: KDE Goal Distribution
# ======================================================================
# KDEGoalDistribution is now imported from fast_kde module


# ======================================================================
# Layer 4: Mixture Monte Carlo
# ======================================================================
# MixtureMonteCarlo is now imported from fast_mixture_mc module


# ======================================================================
# Layer 5: Contextual Adjustment Layer
# ======================================================================
class ContextualLayer:
    """Contextual adjustments based on match situation.

    Features:
    - Rest days (fatigue)
    - Home/away form differential
    - Opponent strength
    - Goal-scoring trend (improving/declining)
    """

    def __init__(self):
        self.team_context: Dict[str, Dict] = {}

    def update(self, team: str, goals_scored: float, goals_conceded: float,
               is_home: bool, rest_days: int = 7):
        """Update contextual information for a team."""
        if team not in self.team_context:
            self.team_context[team] = {
                "recent_scores": [],
                "recent_conceded": [],
                "home_scores": [],
                "away_scores": [],
                "rest_days": [],
            }
        ctx = self.team_context[team]
        ctx["recent_scores"].append(goals_scored)
        ctx["recent_conceded"].append(goals_conceded)
        ctx["rest_days"].append(rest_days)
        if is_home:
            ctx["home_scores"].append(goals_scored)
        else:
            ctx["away_scores"].append(goals_scored)

        # Keep only last 20 matches
        for key in ["recent_scores", "recent_conceded", "rest_days"]:
            if len(ctx[key]) > 20:
                ctx[key] = ctx[key][-20:]

    def get_adjustment(self, team: str, opponent_strength: float,
                       is_home: bool) -> float:
        """Get contextual adjustment for expected goals.

        Returns a multiplier for lambda (goals expectation).
        """
        if team not in self.team_context:
            return 1.0

        ctx = self.team_context[team]
        adjustment = 1.0

        # Rest days adjustment
        if ctx["rest_days"]:
            avg_rest = np.mean(ctx["rest_days"][-5:])
            if avg_rest < 4:
                adjustment *= 0.95  # tired team
            elif avg_rest > 8:
                adjustment *= 1.02  # well-rested

        # Goal trend (recent vs older)
        scores = ctx["recent_scores"]
        if len(scores) >= 10:
            recent = np.mean(scores[-5:])
            older = np.mean(scores[-10:-5])
            trend = recent - older
            adjustment *= (1 + trend * 0.05)  # small trend adjustment

        # Home/away form
        if is_home and ctx["home_scores"]:
            home_avg = np.mean(ctx["home_scores"][-5:])
            overall_avg = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            if home_avg > overall_avg:
                adjustment *= 1.03
        elif not is_home and ctx["away_scores"]:
            away_avg = np.mean(ctx["away_scores"][-5:])
            overall_avg = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            if away_avg < overall_avg:
                adjustment *= 0.97

        return adjustment


# ======================================================================
# Layer 6: Ensemble with Adaptive Weights
# ======================================================================
class AdaptiveEnsemble:
    """Ensemble that learns optimal weights for each model layer.

    Weights are learned via walk-forward validation, not hard-coded.
    """

    def __init__(self):
        self.weights: Dict[str, float] = {
            "poisson": 0.35,
            "kde": 0.25,
            "monte_carlo": 0.25,
            "ml": 0.15,
        }
        self.model_performance: Dict[str, List[float]] = {
            "poisson": [],
            "kde": [],
            "monte_carlo": [],
            "ml": [],
        }

    def record_performance(self, model_name: str, log_loss: float):
        """Record model performance for weight learning."""
        if model_name in self.model_performance:
            self.model_performance[model_name].append(log_loss)

    def update_weights(self):
        """Update ensemble weights based on recent performance.

        Better-performing models (lower log loss) get higher weights.
        """
        avg_performances = {}
        for name, losses in self.model_performance.items():
            if losses:
                # Use last 50 observations for recency
                recent = losses[-50:]
                avg_performances[name] = np.mean(recent)

        if not avg_performances:
            return

        # Inverse log-loss weighting (lower log loss = higher weight)
        inv_losses = {}
        for name, ll in avg_performances.items():
            inv_losses[name] = 1.0 / max(ll, 0.01)

        total = sum(inv_losses.values())
        for name in self.weights:
            if name in inv_losses:
                self.weights[name] = inv_losses[name] / total

    def combine(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Combine predictions from all models using adaptive weights."""
        combined = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}

        for model_name, probs in predictions.items():
            if model_name in self.weights and probs:
                w = self.weights[model_name]
                for outcome in OUTCOMES:
                    if outcome in probs:
                        combined[outcome] += w * probs[outcome]

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        return combined


# ======================================================================
# Full Layered Model
# ======================================================================
class LayeredModel:
    """Full layered model combining all candidate techniques.

    Architecture:
    1. Bayesian Team Prior (strength estimation)
    2. EWMA Recency (recent form)
    3. [Parametric Poisson | KDE | Mixture MC] (goal distribution)
    4. Contextual Adjustments (rest, trend, home/away)
    5. Adaptive Ensemble (model combination)
    6. Calibration (probability adjustment)
    """

    def __init__(self, layers: List[str] = None):
        """Initialize with specified active layers.

        Args:
            layers: List of layer names to activate. If None, all layers active.
                    Options: ["bayesian", "ewma", "kde", "mixture_mc", "contextual", "ensemble"]
        """
        self.active_layers = layers or [
            "bayesian", "ewma", "kde", "mixture_mc", "contextual", "ensemble"
        ]

        # Initialize all layers
        self.bayesian = BayesianTeamPrior()
        self.ewma_home = EWMARecency(alpha=0.3)
        self.ewma_away = EWMARecency(alpha=0.3)
        self.kde = FastKDEGoalDistribution(grid_size=256)
        self.mixture_mc = FastMixtureMC(n_simulations=3000)
        self.online_poisson = OnlinePoissonRatings()
        self.contextual = ContextualLayer()
        self.ensemble = AdaptiveEnsemble()

        # Training data storage
        self.team_home_goals: Dict[str, List[float]] = {}
        self.team_away_goals: Dict[str, List[float]] = {}
        self.league_home_goals: List[float] = []
        self.league_away_goals: List[float] = []
        self.elo_ratings: Dict[str, float] = {}
        self.elo_base = 1500.0
        self.elo_k = 20.0
        self.is_trained = False

    def train(self, df: pd.DataFrame, verbose: bool = True):
        """Train the layered model on historical data."""
        if verbose:
            print(f"Training LayeredModel (active layers: {self.active_layers})")

        df = df.sort_values("date").reset_index(drop=True)

        # Initialize Elo
        all_teams = set(df["home_team"]).union(set(df["away_team"]))
        for t in all_teams:
            self.elo_ratings[t] = self.elo_base

        # Process matches chronologically
        for _, row in df.iterrows():
            h, a = row["home_team"], row["away_team"]
            hg, ag = float(row["home_goals"]), float(row["away_goals"])

            # Update Bayesian prior
            if "bayesian" in self.active_layers:
                self.bayesian.update(h, hg, ag)
                self.bayesian.update(a, ag, hg)

            # Update EWMA
            if "ewma" in self.active_layers:
                self.ewma_home.update(h, hg)
                self.ewma_away.update(a, ag)

            # Update contextual
            if "contextual" in self.active_layers:
                self.contextual.update(h, hg, ag, is_home=True)
                self.contextual.update(a, ag, hg, is_home=False)

            # Store for KDE
            if h not in self.team_home_goals:
                self.team_home_goals[h] = []
            if a not in self.team_away_goals:
                self.team_away_goals[a] = []
            self.team_home_goals[h].append(hg)
            self.team_away_goals[a].append(ag)
            self.league_home_goals.append(hg)
            self.league_away_goals.append(ag)

            # Update Online Poisson ratings
            if "online_poisson" in self.active_layers:
                self.online_poisson.update(h, a, hg, ag)

            # Update Elo
            self._update_elo(h, a, hg, ag)

        # Fit KDE
        if "kde" in self.active_layers:
            for team in all_teams:
                self.kde.fit(team,
                             self.team_home_goals.get(team, []),
                             self.team_away_goals.get(team, []))
            self.kde.fit_league(self.league_home_goals, self.league_away_goals)

        self.is_trained = True
        if verbose:
            print(f"  Trained on {len(df)} matches, {len(all_teams)} teams")
            print(f"  Active layers: {self.active_layers}")

    def _update_elo(self, home: str, away: str, hg: float, ag: float):
        """Update Elo ratings."""
        h_r = self.elo_ratings.get(home, self.elo_base)
        a_r = self.elo_ratings.get(away, self.elo_base)
        exp_h = 1 / (1 + 10 ** ((a_r - h_r) / 400))
        actual = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        self.elo_ratings[home] = h_r + self.elo_k * (actual - exp_h)
        self.elo_ratings[away] = a_r + self.elo_k * ((1 - actual) - (1 - exp_h))

    def predict(self, home_team: str, away_team: str, 
                close_odds_home: Optional[float] = None,
                close_odds_away: Optional[float] = None) -> Dict[str, float]:
        """Predict match outcome probabilities using active layers."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        home_elo = self.elo_ratings.get(home_team, self.elo_base)
        away_elo = self.elo_ratings.get(away_team, self.elo_base)

        # Base Poisson predictions from Elo
        lambda_home = 1.6 * np.exp(0.22 * ((home_elo - away_elo) / 400))
        lambda_away = 1.3 * np.exp(-0.22 * ((home_elo - away_elo) / 400))

        # Layer 1: Bayesian strength adjustment
        if "bayesian" in self.active_layers:
            h_str = self.bayesian.get_strength(home_team)
            a_str = self.bayesian.get_strength(away_team)
            # Blend Elo-based lambda with Bayesian strength
            bay_home = h_str["attack"] * (a_str["defense"] / 1.5)
            bay_away = a_str["attack"] * (h_str["defense"] / 1.5)
            lambda_home = 0.5 * lambda_home + 0.5 * bay_home
            lambda_away = 0.5 * lambda_away + 0.5 * bay_away

        # Layer 2: EWMA recency adjustment
        if "ewma" in self.active_layers:
            ewma_h = self.ewma_home.get_ewma(home_team)
            ewma_a = self.ewma_away.get_ewma(away_team)
            # Blend with EWMA
            lambda_home = 0.6 * lambda_home + 0.4 * ewma_h
            lambda_away = 0.6 * lambda_away + 0.4 * ewma_a

        # Layer 3: Contextual adjustments
        if "contextual" in self.active_layers:
            h_adj = self.contextual.get_adjustment(
                home_team, away_elo / self.elo_base, is_home=True)
            a_adj = self.contextual.get_adjustment(
                away_team, home_elo / self.elo_base, is_home=False)
            lambda_home *= h_adj
            lambda_away *= a_adj

        predictions = {}

        # Layer 4a: Parametric Poisson (always computed as baseline)
        p_h, p_d, p_a = self._poisson_score_grid(lambda_home, lambda_away)
        predictions["poisson"] = {"home_win": p_h, "draw": p_d, "away_win": p_a}

        # Layer 4b: KDE (fast binning + FFT)
        if "kde" in self.active_layers:
            home_data = self.team_home_goals.get(home_team, [])
            away_data = self.team_away_goals.get(away_team, [])
            if len(home_data) >= 5 and len(away_data) >= 5:
                kde_result = self.kde.score_grid_probs(home_team, away_team)
                if kde_result is not None:
                    predictions["kde"] = {
                        "home_win": kde_result[0],
                        "draw": kde_result[1],
                        "away_win": kde_result[2],
                    }

        # Layer 4c: Mixture Monte Carlo (vectorized)
        if "mixture_mc" in self.active_layers:
            h_str = self.bayesian.get_strength(home_team)
            a_str = self.bayesian.get_strength(away_team)
            regime_w = self.mixture_mc.detect_regime(
                h_str["strength"], a_str["strength"],
                self.ewma_home.get_ewma(home_team),
                np.mean(self.team_home_goals.get(home_team, [1.5]))
            )
            mc_h, mc_d, mc_a = self.mixture_mc.simulate(
                lambda_home, lambda_away, regime_w)
            predictions["monte_carlo"] = {
                "home_win": mc_h, "draw": mc_d, "away_win": mc_a
            }

        # Layer 4d: Online Poisson (updated every match)
        if "online_poisson" in self.active_layers:
            op_h, op_d, op_a = self.online_poisson.poisson_score_grid(
                home_team, away_team)
            predictions["online_poisson"] = {
                "home_win": op_h, "draw": op_d, "away_win": op_a
            }

        # Layer 5: ML prediction (if trained)
        # For now, use Poisson as ML proxy
        predictions["ml"] = predictions["poisson"]

        # Layer 6: Ensemble combination
        if "ensemble" in self.active_layers and len(predictions) > 1:
            combined = self.ensemble.combine(predictions)
        else:
            combined = predictions.get("poisson", {"home_win": 0.45, "draw": 0.27, "away_win": 0.28})

        return {k: round(v, 4) for k, v in combined.items()}

    def _poisson_score_grid(self, lam_h: float, lam_a: float,
                            max_goals: int = MAX_GOALS) -> Tuple[float, float, float]:
        """Poisson score grid probability calculation (vectorized)."""
        return vectorized_poisson_score_grid(lam_h, lam_a, max_goals)


# ======================================================================
# Walk-Forward Ablation Tournament
# ======================================================================
def run_ablation_tournament(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Run ablation tournament testing each layer incrementally.

    Tests:
    1. Baseline (Poisson only)
    2. + Bayesian
    3. + EWMA
    4. + KDE
    5. + Mixture MC
    6. + Contextual
    7. + Ensemble (all layers)

    Uses walk-forward validation: train on first 60%, test on last 40%.
    """
    if verbose:
        print("=" * 70)
        print("ABLATION TOURNAMENT — Walk-Forward Validation")
        print("=" * 70)

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * 0.6)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if verbose:
        print(f"Train: {len(train_df)} matches, Test: {len(test_df)} matches")

    # Define layer configurations to test
    configs = [
        ("Baseline (Poisson only)", []),
        ("+ Bayesian", ["bayesian"]),
        ("+ EWMA", ["bayesian", "ewma"]),
        ("+ KDE", ["bayesian", "ewma", "kde"]),
        ("+ Mixture MC", ["bayesian", "ewma", "kde", "mixture_mc"]),
        ("+ Contextual", ["bayesian", "ewma", "kde", "mixture_mc", "contextual"]),
        ("Full Ensemble", ["bayesian", "ewma", "kde", "mixture_mc", "contextual", "ensemble"]),
    ]

    results = []

    for config_name, active_layers in configs:
        if verbose:
            print(f"\n--- {config_name} ---")

        # Train model
        model = LayeredModel(layers=active_layers)
        model.train(train_df, verbose=False)

        # Test on holdout
        all_probs = []
        all_true = []

        for _, row in test_df.iterrows():
            try:
                probs = model.predict(row["home_team"], row["away_team"])
                all_probs.append([probs["away_win"], probs["draw"], probs["home_win"]])

                true_map = {"A": 0, "D": 1, "H": 2}
                all_true.append(true_map.get(row["result"], 1))
            except Exception:
                continue

        if not all_probs:
            continue

        probs_arr = np.array(all_probs)
        y_true = np.array(all_true)

        # Compute metrics
        eps = 1e-9
        log_loss = float(-np.mean(np.log(np.clip(probs_arr[np.arange(len(y_true)), y_true], eps, 1))))
        brier = float(np.mean(np.sum((probs_arr - np.eye(3)[y_true]) ** 2, axis=1)))
        accuracy = float(np.mean(np.argmax(probs_arr, axis=1) == y_true))

        # ECE
        from models.calibration import expected_calibration_error
        ece = expected_calibration_error(probs_arr, y_true)

        result = {
            "config": config_name,
            "n_layers": len(active_layers),
            "log_loss": round(log_loss, 4),
            "brier": round(brier, 4),
            "accuracy": round(accuracy, 4),
            "ece": round(ece, 4),
            "n_test": len(y_true),
        }
        results.append(result)

        if verbose:
            print(f"  Log-loss: {log_loss:.4f} | Brier: {brier:.4f} | "
                  f"Acc: {accuracy:.1%} | ECE: {ece:.4f}")

    results_df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION RESULTS")
        print("=" * 70)
        print(results_df.to_string(index=False))

        # Find best
        best_ll = results_df.loc[results_df["log_loss"].idxmin()]
        best_brier = results_df.loc[results_df["brier"].idxmin()]
        print(f"\nBest log-loss: {best_ll['config']} ({best_ll['log_loss']:.4f})")
        print(f"Best Brier: {best_brier['config']} ({best_brier['brier']:.4f})")

    return results_df


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import generate_match_data

    print("Generating synthetic data...")
    df = generate_match_data(800, seed=42)
    print(f"Generated {len(df)} matches")

    # Run ablation tournament
    results = run_ablation_tournament(df, verbose=True)

    print("\n[OK] Layered model ablation tournament complete.")
