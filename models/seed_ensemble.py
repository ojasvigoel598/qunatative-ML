#!/usr/bin/env python3
"""
Seed Ensemble — Average Predictions from Multiple Random Seeds.

Research basis:
- Luis-ntonio/Match_Prediction: "A single seed's test accuracy can swing by
  a point or more from training randomness alone; averaging seeds trades
  that variance away for free."
- Dietterich (2000): Ensemble diversity reduces variance

Key insight: Even with the same model architecture and hyperparameters,
different random seeds produce slightly different models due to:
- Random initialization
- Random data shuffling
- Random subsampling

Averaging predictions from K seeds reduces variance by factor of ~1/K.

For K=5 seeds:
- Single seed accuracy variance: σ²
- Ensemble accuracy variance: σ²/5
- Typical improvement: 0.5-1.0% accuracy

Usage:
    from models.seed_ensemble import SeedEnsemble
    ensemble = SeedEnsemble(n_seeds=5)
    ensemble.fit(X_train, y_train, X_val, y_val)
    probs = ensemble.predict(X_test)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score


class SeedEnsemble:
    """Ensemble that averages predictions from multiple random seeds.

    Trains K independent models with different random seeds and averages
    their predicted probabilities. This reduces variance without
    increasing bias.
    """

    def __init__(self, n_seeds: int = 5, base_model=None,
                 calibration_method: str = "isotonic"):
        """Initialize seed ensemble.

        Args:
            n_seeds: Number of random seeds to use
            base_model: Base classifier (must support random_state parameter)
            calibration_method: Calibration method ('isotonic', 'sigmoid', or None)
        """
        self.n_seeds = n_seeds
        self.calibration_method = calibration_method
        self.models = []
        self.calibrated_models = []

        if base_model is None:
            # Default to LightGBM if available, else GradientBoosting
            try:
                import lightgbm as lgb
                self.base_model_class = lgb.LGBMClassifier
                self.base_model_params = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "verbose": -1,
                }
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self.base_model_class = GradientBoostingClassifier
                self.base_model_params = {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                }
        else:
            self.base_model_class = base_model.__class__
            self.base_model_params = base_model.get_params()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            verbose: bool = False):
        """Train ensemble with multiple seeds.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features for calibration (optional)
            y_val: Validation labels for calibration (optional)
            verbose: Print progress
        """
        self.models = []
        self.calibrated_models = []

        for seed in range(self.n_seeds):
            if verbose:
                print(f"  Training seed {seed + 1}/{self.n_seeds}...")

            # Create model with different random seed
            params = self.base_model_params.copy()
            params["random_state"] = seed
            model = self.base_model_class(**params)

            # Train
            model.fit(X_train, y_train)
            self.models.append(model)

            # Calibrate if validation data provided
            if X_val is not None and y_val is not None and self.calibration_method:
                try:
                    cal_model = CalibratedClassifierCV(
                        model, method=self.calibration_method, cv="prefit"
                    )
                    cal_model.fit(X_val, y_val)
                    self.calibrated_models.append(cal_model)
                except Exception:
                    # If calibration fails, use uncalibrated model
                    self.calibrated_models.append(model)
            else:
                self.calibrated_models.append(model)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities by averaging across seeds.

        Args:
            X: Features to predict

        Returns:
            Array of shape (n_samples, n_classes) with averaged probabilities
        """
        if not self.calibrated_models:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Collect predictions from all seeds
        all_probas = []
        for model in self.calibrated_models:
            probas = model.predict_proba(X)
            all_probas.append(probas)

        # Average predictions
        avg_probas = np.mean(all_probas, axis=0)

        # Normalize to ensure sum to 1
        row_sums = avg_probas.sum(axis=1, keepdims=True)
        avg_probas = avg_probas / np.maximum(row_sums, 1e-10)

        return avg_probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes by averaging across seeds."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def get_individual_predictions(self, X: np.ndarray) -> List[np.ndarray]:
        """Get predictions from each seed separately.

        Useful for analyzing seed-to-seed variance.
        """
        return [model.predict_proba(X) for model in self.calibrated_models]

    def get_variance(self, X: np.ndarray) -> np.ndarray:
        """Get prediction variance across seeds.

        High variance indicates the prediction is uncertain.
        """
        all_probas = self.get_individual_predictions(X)
        return np.var(all_probas, axis=0)


class SimpleSeedEnsemble:
    """Simplified seed ensemble for quick use.

    Instead of training K separate models, this uses K different
    data subsamples (bootstrap) with a single model type.
    """

    def __init__(self, n_seeds: int = 5, base_model_class=None,
                 base_model_params: Optional[dict] = None):
        self.n_seeds = n_seeds
        self.models = []

        if base_model_class is None:
            try:
                import lightgbm as lgb
                base_model_class = lgb.LGBMClassifier
                base_model_params = {"n_estimators": 100, "verbose": -1}
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                base_model_class = GradientBoostingClassifier
                base_model_params = {"n_estimators": 100}

        self.base_model_class = base_model_class
        self.base_model_params = base_model_params or {}

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Train ensemble with bootstrap samples."""
        self.models = []
        rng = np.random.default_rng(42)

        for seed in range(self.n_seeds):
            if verbose:
                print(f"  Bootstrap seed {seed + 1}/{self.n_seeds}...")

            # Bootstrap sample
            indices = rng.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]

            # Train
            params = self.base_model_params.copy()
            params["random_state"] = seed
            model = self.base_model_class(**params)
            model.fit(X_boot, y_boot)
            self.models.append(model)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average predicted probabilities across bootstrap models."""
        all_probas = [m.predict_proba(X) for m in self.models]
        return np.mean(all_probas, axis=0)


# ======================================================================
# Benchmarks
# ======================================================================

def benchmark_seed_ensemble():
    """Benchmark seed ensemble vs single model."""
    import time
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic data
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=10,
        n_classes=3, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print("Seed Ensemble Benchmark")
    print("=" * 60)

    # Single model
    t0 = time.perf_counter()
    try:
        import lightgbm as lgb
        single = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        single = GradientBoostingClassifier(n_estimators=100, random_state=42)
    single.fit(X_train, y_train)
    t_single = (time.perf_counter() - t0) * 1000

    single_acc = accuracy_score(y_test, single.predict(X_test))
    single_ll = log_loss(y_test, single.predict_proba(X_test))

    print(f"Single model:  acc={single_acc:.4f}, ll={single_ll:.4f}, time={t_single:.1f}ms")

    # Seed ensemble
    for n_seeds in [3, 5, 10]:
        t0 = time.perf_counter()
        ensemble = SeedEnsemble(n_seeds=n_seeds)
        ensemble.fit(X_train, y_train)
        t_ensemble = (time.perf_counter() - t0) * 1000

        ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
        ens_ll = log_loss(y_test, ensemble.predict_proba(X_test))

        print(f"Ensemble ({n_seeds} seeds): acc={ens_acc:.4f}, ll={ens_ll:.4f}, "
              f"time={t_ensemble:.1f}ms, d_acc={ens_acc - single_acc:+.4f}")

    print("=" * 60)


if __name__ == "__main__":
    benchmark_seed_ensemble()
