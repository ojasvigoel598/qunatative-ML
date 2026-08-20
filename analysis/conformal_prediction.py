#!/usr/bin/env python3
"""
Conformal Prediction for Match Outcome Probabilities.

Provides distribution-free prediction intervals with finite-sample
coverage guarantees.  Useful for:
1. Quantifying uncertainty in model predictions
2. Identifying matches where the model is uncertain
3. Adjusting bet sizing based on prediction confidence

Based on:
- Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Lei & Wasserman (2014) "Distribution-free prediction sets"

Usage:
    from analysis.conformal_prediction import ConformalPredictor
    cp = ConformalPredictor()
    intervals = cp.compute_intervals(model_probs, y_true_cal)
    coverage = cp.evaluate_coverage(intervals, y_true_test)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ConformalPredictor:
    """Distribution-free conformal prediction for multi-class outcomes.

    Uses split conformal prediction: a calibration set provides nonconformity
    scores, and the test set gets prediction sets with guaranteed coverage.
    """

    def __init__(self, confidence_level: float = 0.90):
        """
        Args:
            confidence_level: desired coverage probability (e.g. 0.90 for 90%).
        """
        self.confidence_level = confidence_level
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile_threshold: Optional[float] = None

    def nonconformity_scores(self, probs: np.ndarray,
                             y_true: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores: 1 - p(true class).

        Lower score means the true class had higher predicted probability
        (more conforming).
        """
        n = len(y_true)
        scores = 1.0 - probs[np.arange(n), y_true]
        return scores

    def calibrate(self, cal_probs: np.ndarray,
                  cal_y: np.ndarray) -> float:
        """Calibrate on a held-out calibration set.

        Computes the quantile threshold that achieves the desired coverage.
        """
        scores = self.nonconformity_scores(cal_probs, cal_y)
        self.calibration_scores = scores

        # Fractional quantile for finite-sample coverage guarantee
        n = len(scores)
        q_level = np.ceil((n + 1) * self.confidence_level) / n
        q_level = min(q_level, 1.0)
        self.quantile_threshold = float(np.quantile(scores, q_level))

        return self.quantile_threshold

    def predict_sets(self, test_probs: np.ndarray) -> list:
        """Generate prediction sets for each test match.

        A prediction set includes all outcomes whose nonconformity score
        is <= the calibrated threshold.
        """
        if self.quantile_threshold is None:
            raise ValueError("Must call calibrate() first")

        n_classes = test_probs.shape[1]
        class_names = ["away_win", "draw", "home_win"]
        sets = []

        for i in range(len(test_probs)):
            scores = 1.0 - test_probs[i]
            included = [class_names[j] for j in range(n_classes)
                        if scores[j] <= self.quantile_threshold]
            sets.append({
                "index": i,
                "prediction_set": included,
                "set_size": len(included),
                "probabilities": {class_names[j]: round(float(test_probs[i, j]), 4)
                                  for j in range(n_classes)},
            })

        return sets

    def evaluate_coverage(self, test_probs: np.ndarray,
                          test_y: np.ndarray) -> Dict:
        """Evaluate empirical coverage and efficiency.

        Returns:
            coverage: fraction of test matches where the true outcome
                is in the prediction set.
            mean_set_size: average size of prediction sets.
            per_class_coverage: coverage broken down by outcome class.
        """
        if self.quantile_threshold is None:
            raise ValueError("Must call calibrate() first")

        n = len(test_y)
        n_classes = test_probs.shape[1]
        class_names = ["away_win", "draw", "home_win"]

        covered = 0
        set_sizes = []
        per_class_total = np.zeros(n_classes)
        per_class_covered = np.zeros(n_classes)

        for i in range(n):
            scores = 1.0 - test_probs[i]
            in_set = scores <= self.quantile_threshold
            set_sizes.append(int(in_set.sum()))

            if in_set[test_y[i]]:
                covered += 1

            for c in range(n_classes):
                per_class_total[c] += 1
                if in_set[c]:
                    per_class_covered[c] += 1

        empirical_coverage = covered / n
        mean_set_size = np.mean(set_sizes)

        # Efficiency: how much smaller are the sets than the full set?
        efficiency = 1.0 - mean_set_size / n_classes

        per_class = {}
        for c, name in enumerate(class_names):
            if per_class_total[c] > 0:
                per_class[name] = {
                    "coverage": round(float(per_class_covered[c] /
                                          per_class_total[c]), 4),
                    "n_matches": int(per_class_total[c]),
                }

        return {
            "nominal_coverage": self.confidence_level,
            "empirical_coverage": round(empirical_coverage, 4),
            "coverage_gap": round(self.confidence_level - empirical_coverage, 4),
            "mean_set_size": round(float(mean_set_size), 4),
            "efficiency": round(float(efficiency), 4),
            "n_test_matches": n,
            "per_class_coverage": per_class,
            "quantile_threshold": round(self.quantile_threshold, 4),
        }

    def uncertainty_adjusted_edge(self, test_probs: np.ndarray,
                                  odds: np.ndarray,
                                  min_set_size: int = 1) -> np.ndarray:
        """Compute uncertainty-adjusted edge for betting decisions.

        Matches where the prediction set includes multiple outcomes are
        less certain and should be bet more conservatively.

        Returns an array of adjusted edges (one per match, for the best
        outcome).
        """
        n_classes = test_probs.shape[1]
        edges = np.zeros(len(test_probs))

        for i in range(len(test_probs)):
            scores = 1.0 - test_probs[i]
            in_set = scores <= self.quantile_threshold
            set_size = int(in_set.sum())

            if set_size > min_set_size:
                # Reduce edge proportionally to set size
                confidence_factor = 1.0 / set_size
                best_class = np.argmax(test_probs[i] * odds[i] - 1.0)
                raw_edge = test_probs[i, best_class] * odds[i, best_class] - 1.0
                edges[i] = raw_edge * confidence_factor
            else:
                best_class = np.argmax(test_probs[i] * odds[i] - 1.0)
                edges[i] = test_probs[i, best_class] * odds[i, best_class] - 1.0

        return edges

    def summary(self) -> str:
        """Human-readable summary of the conformal predictor state."""
        if self.quantile_threshold is None:
            return "ConformalPredictor (uncalibrated)"
        n_cal = len(self.calibration_scores) if self.calibration_scores is not None else 0
        return (f"ConformalPredictor(confidence={self.confidence_level}, "
                f"threshold={self.quantile_threshold:.4f}, "
                f"cal_size={n_cal})")


# ======================================================================
# Split Conformal Pipeline
# ======================================================================
def split_conformal_pipeline(model_probs: np.ndarray,
                             y: np.ndarray,
                             train_idx: np.ndarray,
                             cal_idx: np.ndarray,
                             test_idx: np.ndarray,
                             confidence_level: float = 0.90
                             ) -> Dict:
    """End-to-end split conformal prediction pipeline.

    Args:
        model_probs: (n, 3) predicted probabilities.
        y: (n,) true class indices.
        train_idx: indices for model training.
        cal_idx: indices for calibration.
        test_idx: indices for test evaluation.

    Returns:
        Dict with calibration results, coverage evaluation, and prediction sets.
    """
    cp = ConformalPredictor(confidence_level=confidence_level)

    # Calibrate on calibration set
    cal_probs = model_probs[cal_idx]
    cal_y = y[cal_idx]
    threshold = cp.calibrate(cal_probs, cal_y)

    # Evaluate on test set
    test_probs = model_probs[test_idx]
    test_y = y[test_idx]
    coverage = cp.evaluate_coverage(test_probs, test_y)

    # Generate prediction sets for test
    sets = cp.predict_sets(test_probs)

    return {
        "threshold": threshold,
        "coverage": coverage,
        "prediction_sets": sets,
        "predictor": cp,
    }


if __name__ == "__main__":
    # Self-test with synthetic data
    rng = np.random.default_rng(42)
    n = 500
    n_classes = 3
    y = rng.integers(0, n_classes, n)

    # Simulate well-calibrated probabilities
    probs = np.full((n, n_classes), 0.1)
    probs[np.arange(n), y] = 0.8
    probs += rng.normal(0, 0.05, probs.shape)
    probs = np.clip(probs, 0.01, 0.99)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Split
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = 200
    n_cal = 100
    train_idx = idx[:n_train]
    cal_idx = idx[n_train:n_train + n_cal]
    test_idx = idx[n_train + n_cal:]

    result = split_conformal_pipeline(probs, y, train_idx, cal_idx, test_idx,
                                      confidence_level=0.90)
    cov = result["coverage"]
    print(f"Nominal coverage:   {cov['nominal_coverage']:.2f}")
    print(f"Empirical coverage: {cov['empirical_coverage']:.4f}")
    print(f"Coverage gap:       {cov['coverage_gap']:.4f}")
    print(f"Mean set size:      {cov['mean_set_size']:.2f} / {n_classes}")
    print(f"Efficiency:         {cov['efficiency']:.4f}")
    print(f"[OK] Conformal prediction self-test passed.")
