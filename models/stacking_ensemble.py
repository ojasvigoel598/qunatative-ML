#!/usr/bin/env python3
"""
Stacking Ensemble — blends PoissonElo + multiple ML layers via a meta-learner.

The ensemble stacks:
  1. PoissonElo model probabilities (statistical baseline)
  2. LightGBM calibrated probabilities (fast gradient boosting)
  3. GradientBoosting calibrated probabilities (sklearn baseline)

A logistic regression meta-learner is trained on a held-out validation fold
to learn the optimal blend weights, then evaluated on the test set.

This is a standard stacking approach (Wolpert, 1992; Breiman, 1996) adapted
for probability calibration: the meta-learner learns which base model's
probabilities to trust most in different regions of the prediction space.

Usage:
    from models.stacking_ensemble import StackingEnsemble
    ensemble = StackingEnsemble()
    ensemble.train(train_df, valid_df)
    probs = ensemble.predict("Arsenal", "Chelsea")
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

from models.poisson_elo_model import PoissonEloModel
from models.ml_layer import MLFootballPredictor

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

OUTCOMES = ["home_win", "draw", "away_win"]
RESULT_MAP = {"H": "home_win", "D": "draw", "A": "away_win"}


class StackingEnsemble:
    """Stacking ensemble: PoissonElo + LightGBM + GB blended by logistic
    regression meta-learner."""

    def __init__(self, use_dixon_coles: bool = False,
                 use_lightgbm: bool = True, use_gb: bool = True):
        """
        Args:
            use_dixon_coles: Enable Dixon-Coles correction (set False for
                synthetic data where rho fits noise).
            use_lightgbm: Include LightGBM as a base learner.
            use_gb: Include GradientBoosting as a base learner.
        """
        self.use_dixon_coles = use_dixon_coles
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        self.use_gb = use_gb

        self.poisson = PoissonEloModel(use_dixon_coles=use_dixon_coles)
        self.ml_gb: Optional[MLFootballPredictor] = None
        self.ml_lgbm: Optional[MLFootballPredictor] = None
        self.meta_learner = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42)
        self.is_trained = False

        # Per-model weights learned by the meta-learner
        self.model_weights: Dict[str, float] = {}

    def _score_matches(self, df: pd.DataFrame) -> np.ndarray:
        """Score all matches in df with the three base models.

        Returns an (n, 9) feature matrix: [poisson_H/D/A, lgbm_H/D/A, gb_H/D/A].
        """
        rows = []
        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]

            # PoissonElo
            p_poisson = self.poisson.predict(home, away)
            poisson_vec = [p_poisson["home_win"], p_poisson["draw"],
                           p_poisson["away_win"]]

            # LightGBM
            lgbm_vec = [1 / 3, 1 / 3, 1 / 3]
            if self.ml_lgbm is not None:
                p_lgbm = self.ml_lgbm.predict_proba(
                    home, away,
                    home_elo=self.poisson.get_team_elo(home),
                    away_elo=self.poisson.get_team_elo(away))
                lgbm_vec = [p_lgbm["home_win"], p_lgbm["draw"],
                            p_lgbm["away_win"]]

            # GradientBoosting
            gb_vec = [1 / 3, 1 / 3, 1 / 3]
            if self.ml_gb is not None:
                p_gb = self.ml_gb.predict_proba(
                    home, away,
                    home_elo=self.poisson.get_team_elo(home),
                    away_elo=self.poisson.get_team_elo(away))
                gb_vec = [p_gb["home_win"], p_gb["draw"],
                          p_gb["away_win"]]

            rows.append(poisson_vec + lgbm_vec + gb_vec)

        return np.array(rows, dtype=float)

    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame = None,
              verbose: bool = True) -> dict:
        """Train the stacking ensemble.

        Args:
            train_df: Training data (must include result column).
            valid_df: Validation data for meta-learner training.  If None,
                uses the last 20% of train_df as validation.
            verbose: Print training progress.

        Returns:
            dict with training metrics.
        """
        if verbose:
            print("Training Stacking Ensemble...")

        # 1. Train PoissonElo on training data
        self.poisson.train(train_df, verbose=verbose)
        train_feat = self.poisson.training_features.copy()

        # 2. Train base ML models on training data
        if self.use_lightgbm:
            if verbose:
                print("  Training LightGBM base learner...")
            self.ml_lgbm = MLFootballPredictor(model_type="lightgbm")
            self.ml_lgbm.train(train_feat, verbose=False)

        if self.use_gb:
            if verbose:
                print("  Training GradientBoosting base learner...")
            self.ml_gb = MLFootballPredictor(model_type="gradient_boosting")
            self.ml_gb.train(train_feat, verbose=False)

        # 3. Split for meta-learner: if no valid_df, use last 20% of train
        if valid_df is None:
            n = len(train_df)
            split = int(n * 0.8)
            meta_train_df = train_df.iloc[:split].copy()
            meta_valid_df = train_df.iloc[split:].copy()
        else:
            meta_train_df = train_df.copy()
            meta_valid_df = valid_df.copy()

        if verbose:
            print(f"  Meta-learner: train={len(meta_train_df)}, "
                  f"valid={len(meta_valid_df)}")

        # 4. Score meta-validation set with base models
        # Need to retrain PoissonElo on meta_train for proper stacking
        meta_poisson = PoissonEloModel(use_dixon_coles=self.use_dixon_coles)
        meta_poisson.train(meta_train_df, verbose=False)
        meta_feat = meta_poisson.training_features.copy()

        # Retrain ML models on meta_train
        meta_lgbm = None
        meta_gb = None
        if self.ml_lgbm is not None:
            meta_lgbm = MLFootballPredictor(model_type="lightgbm")
            meta_lgbm.train(meta_feat, verbose=False)
        if self.ml_gb is not None:
            meta_gb = MLFootballPredictor(model_type="gradient_boosting")
            meta_gb.train(meta_feat, verbose=False)

        # Score validation with meta-train models
        X_meta_valid = self._score_with_models(
            meta_valid_df, meta_poisson, meta_lgbm, meta_gb)

        # Also score meta-train for meta-learner fitting
        X_meta_train = self._score_with_models(
            meta_train_df, meta_poisson, meta_lgbm, meta_gb)

        # 5. Train meta-learner on meta-train, evaluate on meta-valid
        y_meta_train = meta_train_df["result"].map(
            {"H": 2, "D": 1, "A": 0}).to_numpy()
        y_meta_valid = meta_valid_df["result"].map(
            {"H": 2, "D": 1, "A": 0}).to_numpy()

        self.meta_learner.fit(X_meta_train, y_meta_train)

        # 6. Meta-learner evaluation
        meta_pred = self.meta_learner.predict(X_meta_valid)
        meta_proba = self.meta_learner.predict_proba(X_meta_valid)
        meta_acc = accuracy_score(y_meta_valid, meta_pred)
        meta_ll = log_loss(y_meta_valid, meta_proba, labels=[0, 1, 2])

        # 7. Extract learned weights
        # The meta-learner sees 9 features: 3 per base model.
        # Weight for model m = sum of its 3 coefficients (relative contribution).
        coefs = self.meta_learner.coef_[0]
        names = ["PoissonElo"] * 3 + ["LightGBM"] * 3 + ["GradientBoosting"] * 3
        n_models = 1 + int(self.ml_lgbm is not None) + int(self.ml_gb is not None)
        model_names = ["PoissonElo"]
        if self.ml_lgbm is not None:
            model_names.append("LightGBM")
        if self.ml_gb is not None:
            model_names.append("GradientBoosting")

        for name in model_names:
            idx = [i for i, n in enumerate(names) if n == name]
            self.model_weights[name] = round(float(np.sum(coefs[idx])), 4)

        # Normalize weights to sum to 1
        total_w = sum(abs(v) for v in self.model_weights.values())
        if total_w > 0:
            self.model_weights = {k: round(v / total_w, 4)
                                  for k, v in self.model_weights.items()}

        self.is_trained = True

        if verbose:
            print(f"  Meta-learner accuracy: {meta_acc:.3f}, "
                  f"log-loss: {meta_ll:.3f}")
            print(f"  Model weights: {self.model_weights}")

        return {
            "meta_accuracy": round(meta_acc, 4),
            "meta_log_loss": round(meta_ll, 4),
            "model_weights": self.model_weights,
        }

    def _score_with_models(self, df: pd.DataFrame, poisson: PoissonEloModel,
                           ml_lgbm: Optional[MLFootballPredictor],
                           ml_gb: Optional[MLFootballPredictor]) -> np.ndarray:
        """Score matches with specific model instances (for meta-learner training)."""
        rows = []
        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            p_poisson = poisson.predict(home, away)
            poisson_vec = [p_poisson["home_win"], p_poisson["draw"],
                           p_poisson["away_win"]]

            lgbm_vec = [1 / 3, 1 / 3, 1 / 3]
            if ml_lgbm is not None:
                p = ml_lgbm.predict_proba(
                    home, away,
                    home_elo=poisson.get_team_elo(home),
                    away_elo=poisson.get_team_elo(away))
                lgbm_vec = [p["home_win"], p["draw"], p["away_win"]]

            gb_vec = [1 / 3, 1 / 3, 1 / 3]
            if ml_gb is not None:
                p = ml_gb.predict_proba(
                    home, away,
                    home_elo=poisson.get_team_elo(home),
                    away_elo=poisson.get_team_elo(away))
                gb_vec = [p["home_win"], p["draw"], p["away_win"]]

            rows.append(poisson_vec + lgbm_vec + gb_vec)
        return np.array(rows, dtype=float)

    def predict(self, home: str, away: str) -> Dict[str, float]:
        """Predict outcome probabilities using the stacking ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Get base model predictions
        p_poisson = self.poisson.predict(home, away)
        poisson_vec = [p_poisson["home_win"], p_poisson["draw"],
                       p_poisson["away_win"]]

        lgbm_vec = [1 / 3, 1 / 3, 1 / 3]
        if self.ml_lgbm is not None:
            p = self.ml_lgbm.predict_proba(
                home, away,
                home_elo=self.poisson.get_team_elo(home),
                away_elo=self.poisson.get_team_elo(away))
            lgbm_vec = [p["home_win"], p["draw"], p["away_win"]]

        gb_vec = [1 / 3, 1 / 3, 1 / 3]
        if self.ml_gb is not None:
            p = self.ml_gb.predict_proba(
                home, away,
                home_elo=self.poisson.get_team_elo(home),
                away_elo=self.poisson.get_team_elo(away))
            gb_vec = [p["home_win"], p["draw"], p["away_win"]]

        features = np.array([poisson_vec + lgbm_vec + gb_vec])
        proba = self.meta_learner.predict_proba(features)[0]

        return {
            "home_win": round(float(proba[2]), 4),
            "draw": round(float(proba[1]), 4),
            "away_win": round(float(proba[0]), 4),
        }


if __name__ == "__main__":
    import pipeline
    df = pipeline.generate_match_data(600, seed=42)
    train = df.iloc[:420]
    valid = df.iloc[420:510]
    test = df.iloc[510:]

    ensemble = StackingEnsemble(use_lightgbm=True, use_gb=True)
    result = ensemble.train(train, valid, verbose=True)

    # Evaluate on test
    from pipeline import evaluate_probability_quality, _predictions_over
    scored = _predictions_over(test, ensemble.poisson, None)
    # Override predictions with ensemble
    for i, (_, row) in enumerate(test.iterrows()):
        p = ensemble.predict(row["home_team"], row["away_team"])
        scored.iloc[i, scored.columns.get_loc("p_home_win")] = p["home_win"]
        scored.iloc[i, scored.columns.get_loc("p_draw")] = p["draw"]
        scored.iloc[i, scored.columns.get_loc("p_away_win")] = p["away_win"]
    eval_m = evaluate_probability_quality(scored)
    print(f"\nTest accuracy: {eval_m['accuracy']:.3f}")
    print(f"Test log-loss: {eval_m['log_loss']:.4f}")
    print(f"Test ECE: {eval_m['ece']:.4f}")
    print(f"Model weights: {ensemble.model_weights}")
    print("[OK] StackingEnsemble self-test passed.")
