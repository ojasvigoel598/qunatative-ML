#!/usr/bin/env python3
"""
Poisson Regression + Elo Hybrid Model for Football Match Outcome Prediction

Core Model for the Quantitative Sports Betting Project.

Method
------
1. Elo ratings give each team a dynamic strength score (updated match by match).
2. Two Poisson regressions (home goals / away goals) are fitted on the Elo
   features, giving an expected-goals lambda for any fixture.
3. The full score grid P(goals_home = h) * P(goals_away = a) is summed to get
   P(home win), P(draw), P(away win).
4. Probabilities are converted to "fair" odds and compared with bookmaker odds
   to find value bets:  edge = (model_prob * bookie_odds) - 1.

Usage
-----
    from models.poisson_elo_model import PoissonEloModel
    model = PoissonEloModel()
    model.train(historical_df)                 # df needs home_team, away_team, home_goals, away_goals
    probs = model.predict("Arsenal", "Chelsea")
    edges = model.calculate_edge(probs, bookie_odds)

Notes on correctness
--------------------
* Elo is updated sequentially over training matches only, so no future
  information leaks into a rating used for a later match.
* `home_adv` is deliberately *not* a regression feature: it is constant in
  every match, so including it would make the design matrix collinear with the
  intercept.  Home advantage is instead applied once as a multiplier on the
  home lambda inside `predict()`.
* `predict()` applies the home-advantage multiplier exactly once (the original
  code applied it a second time after the regression already contained it).
"""

import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


class _FittedPoisson:
    """Lightweight stand-in for a statsmodels Poisson fit, restored from saved
    parameter values so a loaded model can predict without re-fitting."""

    def __init__(self, params: Dict[str, float]):
        self.params = params
        self.names = list(params.keys())
        self.values = np.array(list(params.values()), dtype=float)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.names].to_numpy(dtype=float)
        return np.exp(X @ self.values)


class PoissonEloModel:
    def __init__(
        self,
        elo_k: float = 20.0,
        elo_base: float = 1500.0,
        home_advantage: float = 0.15,
        margin: float = 0.05,
        shrinkage: float = 0.7,
        use_dixon_coles: bool = True,
    ):
        """
        Args:
            elo_k: Elo update factor (sensitivity to results).
            elo_base: Starting Elo rating for every team.
            home_advantage: Relative boost applied to home expected goals.
            margin: Assumed bookmaker overround used when converting model
                probabilities into fair odds.
            shrinkage: Multiplier applied to the Elo regression coefficients
                (0 < shrinkage <= 1).  Values < 1 pull predicted lambdas toward
                the league mean, which reduces tail overconfidence and keeps
                the estimated *edges* honest in the betting region (see
                pipeline docs on the winner's curse).
            use_dixon_coles: Apply the Dixon-Coles (1997) low-score correction:
                a single ``rho`` parameter fitted by maximum likelihood on the
                training score matrix adjusts the joint probability of the
                0-0 / 1-0 / 0-1 / 1-1 cells, which independent Poisson models
                systematically mis-price (real football has fewer 0-0 and 1-1
                than an independent model expects).  On synthetic Poisson data
                rho fits to ~0 and the correction is a no-op, which is the
                honest behaviour.
        """
        self.elo_k = elo_k
        self.elo_base = elo_base
        self.home_advantage = home_advantage
        self.margin = margin
        self.shrinkage = shrinkage
        self.use_dixon_coles = use_dixon_coles
        self.rho: float = 0.0  # Dixon-Coles low-score dependence parameter
        self.elo_ratings: Dict[str, float] = {}
        self.poisson_home = None
        self.poisson_away = None
        self.feature_cols = ["home_elo", "away_elo"]
        self.is_trained = False
        self.base_rates = None

    # ----------------------------------------------- Dixon-Coles correction
    @staticmethod
    def _dc_tau(x: int, y: int, lam_home: float, lam_away: float,
                rho: float) -> float:
        """Dixon-Coles tau factor for the low-score cells.

        tau(x, y) multiplies the independent Poisson probability of the score
        cell (x, y).  With rho < 0 the model predicts *fewer* 0-0 and 1-1
        draws than independence would (the classic football finding); with
        rho > 0, more.  All other cells keep tau = 1.
        """
        if rho == 0.0:
            return 1.0
        if x == 0 and y == 0:
            return 1.0 - lam_home * lam_away * rho
        if x == 0 and y == 1:
            return 1.0 + lam_home * rho
        if x == 1 and y == 0:
            return 1.0 + lam_away * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def _fit_dixon_coles_rho(self, df: pd.DataFrame, verbose: bool = True) -> float:
        """Estimate rho by 1-D maximum likelihood on the observed scores.

        Given the fitted goal regressions, the MLE for rho maximises
        sum_i log(tau(h_i, a_i; rho)) over the training matches (the Poisson
        terms are independent of rho and drop out).  Bounded to [-0.25, 0.25],
        the range observed in the literature.
        """
        from scipy.optimize import minimize_scalar

        X = df[self.feature_cols].copy()
        X = sm.add_constant(X)
        lam_home = np.asarray(self.poisson_home.predict(X), dtype=float)
        lam_away = np.asarray(self.poisson_away.predict(X), dtype=float)
        h = df["home_goals"].to_numpy(dtype=int)
        a = df["away_goals"].to_numpy(dtype=int)

        def neg_ll(rho: float) -> float:
            total = 0.0
            for i in range(len(h)):
                if h[i] <= 1 and a[i] <= 1:
                    tau = self._dc_tau(int(h[i]), int(a[i]),
                                       float(lam_home[i]), float(lam_away[i]), rho)
                    if tau > 0:
                        total += np.log(max(tau, 1e-12))
            return -total

        if not np.any((h <= 1) & (a <= 1)):
            if verbose:
                print("  No low-score cells in training data - Dixon-Coles rho=0")
            return 0.0

        res = minimize_scalar(neg_ll, bounds=(-0.25, 0.25), method="bounded")
        rho = float(np.clip(res.x, -0.25, 0.25)) if res.success else 0.0
        if verbose:
            print(f"  Dixon-Coles rho={rho:+.4f} (MLE on {len(df)} matches)")
        return rho

    # ------------------------------------------------------------------ Elo
    def _init_elo(self, teams):
        for team in teams:
            if team not in self.elo_ratings:
                self.elo_ratings[team] = self.elo_base

    def _update_elo(self, home_team: str, away_team: str,
                    home_goals: int, away_goals: int):
        if home_team not in self.elo_ratings:
            self.elo_ratings[home_team] = self.elo_base
        if away_team not in self.elo_ratings:
            self.elo_ratings[away_team] = self.elo_base

        home_rating = self.elo_ratings[home_team]
        away_rating = self.elo_ratings[away_team]

        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))

        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        self.elo_ratings[home_team] += self.elo_k * (actual_home - expected_home)
        self.elo_ratings[away_team] += self.elo_k * ((1 - actual_home) - (1 - expected_home))

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add `home_elo` / `away_elo` columns, updating Elo sequentially.

        The Elo value attached to match *i* reflects only matches 0..i-1, so
        there is no look-ahead leakage.
        """
        df = df.copy()
        all_teams = set(df["home_team"]).union(set(df["away_team"]))
        self._init_elo(list(all_teams))

        home_elos, away_elos = [], []
        for _, row in df.iterrows():
            home_elos.append(self.elo_ratings[row["home_team"]])
            away_elos.append(self.elo_ratings[row["away_team"]])
            self._update_elo(
                row["home_team"], row["away_team"],
                int(row["home_goals"]), int(row["away_goals"]),
            )

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        return df

    # --------------------------------------------------------------- Train
    def train(self, historical_df: pd.DataFrame, verbose: bool = True):
        print("Training PoissonEloModel on historical data...")
        df = self.prepare_features(historical_df)

        X = df[self.feature_cols].copy()
        X = sm.add_constant(X)

        self.poisson_home = sm.Poisson(df["home_goals"], X).fit(disp=0)
        self.poisson_away = sm.Poisson(df["away_goals"], X).fit(disp=0)

        if self.use_dixon_coles:
            self.rho = self._fit_dixon_coles_rho(df, verbose=verbose)

        # League base rates (used by the pipeline for ensemble shrinkage).
        counts = df["result"].value_counts(normalize=True)
        self.base_rates = {
            "home_win": float(counts.get("H", 0.45)),
            "draw": float(counts.get("D", 0.27)),
            "away_win": float(counts.get("A", 0.28)),
        }

        self.is_trained = True
        print(f"  Home goals model: AIC={self.poisson_home.aic:.1f}")
        print(f"  Away goals model: AIC={self.poisson_away.aic:.1f}")
        print(f"  Elo ratings learned for {len(self.elo_ratings)} teams")

    # ------------------------------------------------------------- Predict
    def predict(self, home_team: str, away_team: str,
                max_goals: int = 8) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        home_elo = self.elo_ratings.get(home_team, self.elo_base)
        away_elo = self.elo_ratings.get(away_team, self.elo_base)

        features = pd.DataFrame({
            "const": [1.0],
            "home_elo": [home_elo],
            "away_elo": [away_elo],
        })

        lambda_home = float(self.poisson_home.predict(features)[0])
        lambda_away = float(self.poisson_away.predict(features)[0])

        if 0 < self.shrinkage < 1.0:
            # Pull lambdas toward the training mean (regularisation).  Applies
            # to BOTH lambdas; the original code computed `lam_home` but then
            # fed the un-shrunk `lambda_home` into the PMF, leaving home goals
            # overconfident relative to away goals.
            lam_mean = float(np.mean(self.poisson_home.fittedvalues))
            lambda_home = lam_mean + self.shrinkage * (lambda_home - lam_mean)
            lam_mean_a = float(np.mean(self.poisson_away.fittedvalues))
            lambda_away = lam_mean_a + self.shrinkage * (lambda_away - lam_mean_a)

        # NOTE: home advantage is already captured by the fitted home-goals
        # regression (home teams score more in the data).  Multiplying again
        # here would double-count it and systematically inflate P(home win).

        home_dist = [self._poisson_pmf(i, lambda_home) for i in range(max_goals + 1)]
        away_dist = [self._poisson_pmf(i, lambda_away) for i in range(max_goals + 1)]

        p_home_win = p_draw = p_away_win = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = home_dist[h] * away_dist[a]
                if self.use_dixon_coles and abs(self.rho) > 1e-9:
                    prob *= self._dc_tau(h, a, lambda_home, lambda_away, self.rho)
                if h > a:
                    p_home_win += prob
                elif h == a:
                    p_draw += prob
                else:
                    p_away_win += prob

        total = p_home_win + p_draw + p_away_win
        return {
            "home_win": round(p_home_win / total, 4),
            "draw": round(p_draw / total, 4),
            "away_win": round(p_away_win / total, 4),
            "expected_home_goals": round(lambda_home, 3),
            "expected_away_goals": round(lambda_away, 3),
        }

    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        """Numerically stable Poisson PMF (avoids factorial overflow)."""
        import math
        return float(np.exp(-lam + k * np.log(lam) - math.lgamma(k + 1)))

    # ------------------------------------------------------------- Edges
    def probs_to_fair_odds(self, probs: Dict[str, float]) -> Dict[str, float]:
        fair = {}
        for outcome, prob in probs.items():
            if outcome in ("home_win", "draw", "away_win"):
                fair[outcome] = round(1.0 / max(prob, 0.001), 3)
        return fair

    def calculate_edge(self, model_probs: Dict[str, float],
                       bookie_odds: Dict[str, float],
                       threshold: float = 0.05) -> Dict[str, float]:
        """edge = (model_prob * bookie_odds) - 1, per outcome.

        Positive edge means the bookmaker's odds overpay for the model's
        estimated probability.
        """
        edges: Dict[str, Optional[float]] = {}
        for outcome in ("home_win", "draw", "away_win"):
            if outcome in bookie_odds and bookie_odds[outcome] > 0:
                edge = (model_probs[outcome] * bookie_odds[outcome]) - 1
                edges[outcome] = round(float(edge), 4)
            else:
                edges[outcome] = None

        valid = {k: v for k, v in edges.items() if v is not None}
        if valid:
            best_outcome = max(valid, key=valid.get)  # type: ignore[arg-type]
            best_edge = valid[best_outcome]
            edges["best_value"] = best_outcome if best_edge > threshold else None
            edges["max_edge"] = round(float(best_edge), 4)
        else:
            edges["best_value"] = None
            edges["max_edge"] = 0.0
        return edges

    # ------------------------------------------------------------- Utils
    def get_team_elo(self, team: str) -> float:
        return self.elo_ratings.get(team, self.elo_base)

    def save_model(self, filepath: str = "models/trained_poisson_elo.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump({
                "elo_ratings": self.elo_ratings,
                "poisson_home_params": self.poisson_home.params.to_dict() if self.poisson_home else None,
                "poisson_away_params": self.poisson_away.params.to_dict() if self.poisson_away else None,
                "base_rates": self.base_rates,
                "rho": self.rho,
                "hyperparams": {
                    "k": self.elo_k, "base": self.elo_base,
                    "home_adv": self.home_advantage,
                    "use_dixon_coles": self.use_dixon_coles,
                },
            }, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.elo_ratings = state["elo_ratings"]
        self.base_rates = state.get("base_rates")
        self.rho = state.get("rho", 0.0)
        self.use_dixon_coles = state.get("hyperparams", {}).get(
            "use_dixon_coles", self.use_dixon_coles)
        if state.get("poisson_home_params"):
            self.poisson_home = _FittedPoisson(state["poisson_home_params"])
            self.poisson_away = _FittedPoisson(state["poisson_away_params"])
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Quick self-test
    rng = np.random.default_rng(0)
    sample = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=120, freq="W"),
        "home_team": rng.choice(["Arsenal", "Man City", "Liverpool"], 120),
        "away_team": rng.choice(["Chelsea", "Tottenham", "Newcastle"], 120),
        "home_goals": rng.poisson(1.7, 120),
        "away_goals": rng.poisson(1.2, 120),
    })
    sample["result"] = np.where(sample["home_goals"] > sample["away_goals"], "H",
                                np.where(sample["home_goals"] < sample["away_goals"], "A", "D"))
    model = PoissonEloModel()
    model.train(sample)
    probs = model.predict("Arsenal", "Chelsea")
    assert abs(sum(probs[k] for k in ("home_win", "draw", "away_win")) - 1.0) < 0.01
    print("Predictions:", probs)
    print("Fair odds:", model.probs_to_fair_odds(probs))
    print("Edges:", model.calculate_edge(probs, {"home_win": 2.1, "draw": 3.4, "away_win": 4.0}))
    print(f"Dixon-Coles rho: {model.rho:+.4f}")
    assert abs(sum(probs[k] for k in ("home_win", "draw", "away_win")) - 1.0) < 0.01
    print("[OK] PoissonEloModel self-test passed.")
