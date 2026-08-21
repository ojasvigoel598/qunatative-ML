#!/usr/bin/env python3
"""
Temperature Scaling — post-hoc probability calibration (Guo et al., 2017).

The quantbet project documented this as critical: "A model that says 44%
and delivers 11% turns every EV calculation into fiction."

Temperature scaling learns a single parameter T that rescales logits:
  p_calibrated = softmax(logits / T)

If T > 1: model is overconfident (common for tree ensembles)
If T < 1: model is underconfident
If T = 1: model is well-calibrated

This is fitted on a chronological validation set (never on test data).

Research basis:
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
- quantbet: "Post-hoc temperature scaling (Guo et al., 2017)"
- Niculescu-Mizil & Caruana (2005): calibration is essential for betting

Usage:
    from models.temperature_scaling import TemperatureScaler
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)
    calibrated_probs = scaler.transform(test_logits)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from typing import Tuple


class TemperatureScaler:
    """Post-hoc temperature scaling for probability calibration.

    Learns a single temperature parameter T on a validation set,
    then applies it to any set of logits to produce calibrated probabilities.

    The temperature is fitted by minimizing negative log-likelihood
    (cross-entropy) on the validation set.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            verbose: bool = False) -> float:
        """Fit temperature on validation data.

        Args:
            logits: (n, 3) raw logits from the model (before softmax)
            labels: (n,) class indices {0, 1, 2}
            verbose: print fitting info

        Returns:
            Fitted temperature value
        """
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)

        def nll(T):
            """Negative log-likelihood at temperature T."""
            scaled = logits / T
            # Numerically stable softmax
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=1, keepdims=True)
            eps = 1e-9
            log_probs = np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1))
            return -float(np.mean(log_probs))

        # Search for optimal T in reasonable range
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = float(result.x)
        self.is_fitted = True

        if verbose:
            nll_before = nll(1.0)
            nll_after = nll(self.temperature)
            print(f"  Temperature scaling: T={self.temperature:.3f} "
                  f"(NLL: {nll_before:.4f} -> {nll_after:.4f})")

        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits: (n, 3) raw logits

        Returns:
            (n, 3) calibrated probabilities (sum to 1 per row)
        """
        if not self.is_fitted:
            raise ValueError("TemperatureScaler not fitted. Call fit() first.")

        logits = np.asarray(logits, dtype=float)
        scaled = logits / self.temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def get_params(self) -> dict:
        """Get scaler parameters for serialization."""
        return {"temperature": self.temperature, "is_fitted": self.is_fitted}


def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits (inverse of softmax).

    Args:
        probs: (n, 3) probability matrix

    Returns:
        (n, 3) logits
    """
    eps = 1e-9
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.log(probs)


if __name__ == "__main__":
    # Self-test: overconfident model should be corrected
    rng = np.random.default_rng(42)
    n = 1000

    # Generate true labels
    true_probs = np.array([0.6, 0.25, 0.15])  # home-heavy
    y = rng.choice(3, n, p=true_probs)

    # Simulate overconfident model logits (pushed away from uniform)
    logits = np.zeros((n, 3))
    logits[np.arange(n), y] = 2.0  # confident on true class
    logits += rng.normal(0, 0.5, (n, 3))

    # Raw probs (before calibration)
    shifted = logits - logits.max(axis=1, keepdims=True)
    raw_probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    # Fit temperature on first half, test on second half
    split = n // 2
    scaler = TemperatureScaler()
    scaler.fit(logits[:split], y[:split], verbose=True)

    # Apply to test set
    cal_probs = scaler.transform(logits[split:])

    # Compare calibration
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.calibration import expected_calibration_error, brier_score

    raw_test = raw_probs[split:]
    y_test = y[split:]

    raw_ece = expected_calibration_error(raw_test, y_test)
    cal_ece = expected_calibration_error(cal_probs, y_test)
    raw_brier = brier_score(raw_test, y_test)
    cal_brier = brier_score(cal_probs, y_test)

    print(f"\n  Raw ECE:      {raw_ece:.4f}")
    print(f"  Calibrated ECE: {cal_ece:.4f}")
    print(f"  Raw Brier:    {raw_brier:.4f}")
    print(f"  Calibrated Brier: {cal_brier:.4f}")

    assert cal_ece <= raw_ece + 0.01, "Calibration should not make ECE worse"
    print("\n[OK] Temperature scaling self-test passed.")
