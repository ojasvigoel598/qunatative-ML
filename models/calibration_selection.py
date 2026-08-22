#!/usr/bin/env python3
"""
Calibration-Based Model Selection

Based on Walsh & Joshi (2024): "Machine learning for sports betting:
should model selection be based on accuracy or calibration?"

Key finding: Selecting models based on calibration (ECE) gives +34.69% ROI
vs -35.17% for accuracy-based selection.

This module implements calibration-based model selection for our betting system.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelCandidate:
    """A model candidate with its calibration metrics."""
    name: str
    model: object
    ece: float  # Expected Calibration Error (lower is better)
    brier: float  # Brier score (lower is better)
    log_loss: float  # Log loss (lower is better)
    accuracy: float  # Accuracy (higher is better)
    n_bins: int = 10  # Number of calibration bins


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match actual outcomes.
    Lower ECE = better calibration = more profitable betting.
    
    Args:
        y_true: True labels (0, 1, 2 for away, draw, home)
        y_prob: Predicted probabilities (N, 3)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for class_idx in range(y_prob.shape[1]):
        class_probs = y_prob[:, class_idx]
        class_true = (y_true == class_idx).astype(float)
        
        for bin_idx in range(n_bins):
            lower = bin_boundaries[bin_idx]
            upper = bin_boundaries[bin_idx + 1]
            
            mask = (class_probs > lower) & (class_probs <= upper)
            if mask.sum() > 0:
                bin_accuracy = class_true[mask].mean()
                bin_confidence = class_probs[mask].mean()
                bin_weight = mask.sum() / len(y_true)
                
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Maximum Calibration Error (MCE).
    
    MCE measures the worst-case calibration error across all bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    
    for class_idx in range(y_prob.shape[1]):
        class_probs = y_prob[:, class_idx]
        class_true = (y_true == class_idx).astype(float)
        
        for bin_idx in range(n_bins):
            lower = bin_boundaries[bin_idx]
            upper = bin_boundaries[bin_idx + 1]
            
            mask = (class_probs > lower) & (class_probs <= upper)
            if mask.sum() > 0:
                bin_accuracy = class_true[mask].mean()
                bin_confidence = class_probs[mask].mean()
                mce = max(mce, abs(bin_accuracy - bin_confidence))
    
    return mce


def calibration_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """Compute calibration slope.
    
    Perfect calibration has slope = 1.0.
    Slope < 1.0 = overconfident.
    Slope > 1.0 = underconfident.
    """
    from scipy.special import logit
    
    # Use home win probability as proxy
    probs = np.clip(y_prob[:, 2], 1e-7, 1 - 1e-7)
    logits = logit(probs)
    
    # Simple linear regression
    X = np.column_stack([logits, np.ones(len(logits))])
    y = ((y_true == 2).astype(float))
    
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return beta[0]
    except:
        return 1.0


def calibration_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """Compute calibration intercept."""
    from scipy.special import logit
    
    probs = np.clip(y_prob[:, 2], 1e-7, 1 - 1e-7)
    logits = logit(probs)
    
    X = np.column_stack([logits, np.ones(len(logits))])
    y = ((y_true == 2).astype(float))
    
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return beta[1]
    except:
        return 0.0


def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """Compute Brier score (lower is better)."""
    n_classes = y_prob.shape[1]
    y_one_hot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))


def log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """Compute log loss (lower is better)."""
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(np.log(y_prob_clipped[np.arange(len(y_true)), y_true]))


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Full calibration evaluation.
    
    Returns:
        Dictionary with all calibration metrics
    """
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins),
        "cal_slope": calibration_slope(y_true, y_prob),
        "cal_intercept": calibration_intercept(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "accuracy": float(np.mean(np.argmax(y_prob, axis=1) == y_true)),
    }


def select_model_by_calibration(
    candidates: List[ModelCandidate],
    selection_metric: str = "ece"
) -> ModelCandidate:
    """Select the best model based on calibration.
    
    This is the KEY technique from Walsh & Joshi (2024).
    Instead of selecting by accuracy or log-loss, select by calibration.
    
    Args:
        candidates: List of model candidates with metrics
        selection_metric: Which metric to use for selection
                         "ece" (default) - Expected Calibration Error
                         "brier" - Brier score
                         "cal_slope" - Calibration slope (closer to 1.0 is better)
    
    Returns:
        Best model candidate
    """
    if selection_metric == "ece":
        # Lower ECE is better
        return min(candidates, key=lambda c: c.ece)
    elif selection_metric == "brier":
        # Lower Brier is better
        return min(candidates, key=lambda c: c.brier)
    elif selection_metric == "cal_slope":
        # Closer to 1.0 is better
        return min(candidates, key=lambda c: abs(c.cal_slope - 1.0))
    elif selection_metric == "log_loss":
        # Lower log loss is better
        return min(candidates, key=lambda c: c.log_loss)
    elif selection_metric == "accuracy":
        # Higher accuracy is better
        return max(candidates, key=lambda c: c.accuracy)
    else:
        raise ValueError(f"Unknown selection metric: {selection_metric}")


def select_model_by_composite(
    candidates: List[ModelCandidate],
    weights: Optional[Dict[str, float]] = None
) -> ModelCandidate:
    """Select model using weighted combination of metrics.
    
    Default weights favor calibration over accuracy:
    - ECE: 40% (calibration is most important)
    - Brier: 20% (combines calibration and accuracy)
    - Log-loss: 20% (sharpness)
    - Accuracy: 20% (correctness)
    
    Args:
        candidates: List of model candidates
        weights: Optional custom weights
    
    Returns:
        Best model candidate
    """
    if weights is None:
        weights = {
            "ece": 0.40,      # Calibration most important
            "brier": 0.20,    # Combined metric
            "log_loss": 0.20, # Sharpness
            "accuracy": 0.20, # Correctness
        }
    
    # Normalize each metric to [0, 1]
    metrics = {
        "ece": [c.ece for c in candidates],
        "brier": [c.brier for c in candidates],
        "log_loss": [c.log_loss for c in candidates],
        "accuracy": [c.accuracy for c in candidates],
    }
    
    normalized = {}
    for metric, values in metrics.items():
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            if metric == "accuracy":
                # Higher is better
                normalized[metric] = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                # Lower is better
                normalized[metric] = [(max_val - v) / (max_val - min_val) for v in values]
        else:
            normalized[metric] = [0.5] * len(values)
    
    # Compute weighted scores
    scores = []
    for i in range(len(candidates)):
        score = sum(
            weights.get(m, 0) * normalized[m][i]
            for m in weights
        )
        scores.append(score)
    
    # Return best
    best_idx = np.argmax(scores)
    return candidates[best_idx]


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model"
) -> str:
    """Generate a calibration report.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of the model
    
    Returns:
        Formatted report string
    """
    metrics = evaluate_calibration(y_true, y_prob)
    
    report = f"""
{'='*60}
CALIBRATION REPORT: {model_name}
{'='*60}

Calibration Metrics:
  ECE (Expected Calibration Error): {metrics['ece']:.4f}
  MCE (Maximum Calibration Error):  {metrics['mce']:.4f}
  Calibration Slope:                {metrics['cal_slope']:.4f}
  Calibration Intercept:            {metrics['cal_intercept']:.4f}

Performance Metrics:
  Brier Score:     {metrics['brier']:.4f}
  Log Loss:        {metrics['log_loss']:.4f}
  Accuracy:        {metrics['accuracy']:.4f}

Interpretation:
  ECE < 0.05: Good calibration
  ECE < 0.02: Excellent calibration
  Slope = 1.0: Perfect calibration
  Slope < 1.0: Overconfident (common in ML)
  Slope > 1.0: Underconfident
{'='*60}
"""
    return report


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # True labels
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.25, 0.45])
    
    # Well-calibrated predictions
    y_prob_calibrated = np.random.dirichlet([2, 2, 2], size=n_samples)
    
    # Overconfident predictions (common in ML)
    y_prob_overconfident = np.random.dirichlet([0.5, 0.5, 0.5], size=n_samples)
    
    print("Evaluating well-calibrated model:")
    print(calibration_report(y_true, y_prob_calibrated, "Calibrated Model"))
    
    print("\nEvaluating overconfident model:")
    print(calibration_report(y_true, y_prob_overconfident, "Overconfident Model"))
    
    # Model selection comparison
    candidates = [
        ModelCandidate("Calibrated", None, 
                      expected_calibration_error(y_true, y_prob_calibrated),
                      brier_score(y_true, y_prob_calibrated),
                      log_loss(y_true, y_prob_calibrated),
                      float(np.mean(np.argmax(y_prob_calibrated, axis=1) == y_true))),
        ModelCandidate("Overconfident", None,
                      expected_calibration_error(y_true, y_prob_overconfident),
                      brier_score(y_true, y_prob_overconfident),
                      log_loss(y_true, y_prob_overconfident),
                      float(np.mean(np.argmax(y_prob_overconfident, axis=1) == y_true))),
    ]
    
    print("\nModel Selection Comparison:")
    print(f"  By ECE:       {select_model_by_calibration(candidates, 'ece').name}")
    print(f"  By Brier:     {select_model_by_calibration(candidates, 'brier').name}")
    print(f"  By Accuracy:  {select_model_by_calibration(candidates, 'accuracy').name}")
    print(f"  By Composite: {select_model_by_composite(candidates).name}")
    
    print("\n[OK] Calibration-based model selection complete.")
