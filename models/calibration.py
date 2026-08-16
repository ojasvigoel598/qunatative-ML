#!/usr/bin/env python3
"""
Probability-calibration and market-comparison evaluation helpers.

This module is the shared toolkit for the *model user* view of the system:
it measures whether predicted probabilities are honest (calibration, Brier,
log loss), converts bookmaker odds into implied probabilities, and compares
the model against the market on identical matches.

Class conventions (match the rest of the project):
    * Probability matrices are (n, 3) ordered [away_win, draw, home_win].
    * y_true is a vector of class indices {0: away, 1: draw, 2: home}.

Usage:
    from models.calibration import (expected_calibration_error,
                                    isotonic_fit, isotonic_apply,
                                    compare_to_market, implied_probs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

OUTCOMES = ["away_win", "draw", "home_win"]
RESULT_CLASS = {"A": 0, "D": 1, "H": 2}

# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------
def log_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    """Multiclass log loss of (n, 3) probs against class indices y."""
    probs = np.clip(probs, eps, 1.0)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error vs one-hot)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == y))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray,
                               n_bins: int = 10) -> float:
    """Mean over the three classes of the binary ECE (predicted vs observed).

    ECE is computed per outcome class, then averaged.  A well-calibrated
    model scores near 0; the winner's-curse signature is a positive ECE in
    the betting region (model says 55%, wins 48%).
    """
    eces = []
    for c in range(3):
        p_c = probs[:, c]
        y_c = (y == c).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_sum = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            # last bin inclusive at hi
            mask = (p_c > lo) & (p_c <= hi) if i < n_bins - 1 else \
                (p_c >= lo) & (p_c <= hi)
            if not mask.any():
                continue
            frac_pos = float(y_c[mask].mean())
            mean_p = float(p_c[mask].mean())
            bin_sum += abs(frac_pos - mean_p) * (mask.sum() / len(y_c))
        eces.append(bin_sum)
    return float(np.mean(eces))


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Per-class reliability data: (predicted means, observed frequencies,
    sample counts) for plotting calibration curves."""
    curves = {}
    for c in range(3):
        p_c = probs[:, c]
        y_c = (y == c).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        preds, obs, counts = [], [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (p_c > lo) & (p_c <= hi) if i < n_bins - 1 else \
                (p_c >= lo) & (p_c <= hi)
            if not mask.any():
                continue
            preds.append(float(p_c[mask].mean()))
            obs.append(float(y_c[mask].mean()))
            counts.append(int(mask.sum()))
        curves[OUTCOMES[c]] = (np.array(preds), np.array(obs), np.array(counts))
    return curves


# ---------------------------------------------------------------------------
# Bookmaker implied probabilities
# ---------------------------------------------------------------------------
def implied_probs(odds: dict) -> dict:
    """Normalise 1/odds per outcome to sum to 1 (removes the overround).

    The bookmaker's *true* probability estimate is the implied probability
    after the margin is divided out; this is the fair comparison for the
    model's probabilities.  ``odds`` keys are outcome names, decimal odds.
    """
    inv = {k: 1.0 / float(v) for k, v in odds.items() if v and v > 1.0}
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in inv.items()}


def bookie_probs_matrix(df: pd.DataFrame, home_col: str, draw_col: str,
                        away_col: str) -> np.ndarray:
    """(n, 3) implied-probability matrix [away, draw, home] from odds columns."""
    rows = []
    for _, r in df.iterrows():
        ip = implied_probs({"home_win": r[home_col], "draw": r[draw_col],
                            "away_win": r[away_col]})
        rows.append([ip.get("away_win", 1 / 3), ip.get("draw", 1 / 3),
                     ip.get("home_win", 1 / 3)])
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------
def isotonic_fit(probs: np.ndarray, y: np.ndarray, out_of_sample: bool = True):
    """Fit three per-class isotonic regressions on (probs, y).

    ``out_of_sample=True`` fits each class on all *other* classes' matches
    (the standard trick: the calibration map for class c is learned from
    rows where c did not win, so it is not evaluated on its own fitting
    data).  Returns a list of three IsotonicRegression objects.
    """
    regs = []
    for c in range(3):
        y_c = (y == c).astype(float)
        if out_of_sample:
            fit_mask = y_c == 0  # learn from rows where class c did NOT win
            if fit_mask.sum() == 0:
                fit_mask = np.ones_like(y_c, dtype=bool)
        else:
            fit_mask = np.ones_like(y_c, dtype=bool)
        reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        reg.fit(probs[fit_mask, c], y_c[fit_mask])
        regs.append(reg)
    return regs


def isotonic_apply(probs: np.ndarray, regs) -> np.ndarray:
    """Apply per-class isotonic maps and renormalise rows to sum to 1."""
    out = np.empty_like(probs)
    for c in range(3):
        out[:, c] = regs[c].predict(probs[:, c])
    out = np.clip(out, 1e-6, 1.0)
    return out / out.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Model vs market comparison
# ---------------------------------------------------------------------------
def compare_to_market(model_probs: np.ndarray, market_probs: np.ndarray,
                      y: np.ndarray) -> dict:
    """Head-to-head model vs bookmaker on identical matches.

    Returns log loss, Brier, accuracy and ECE for both, plus the count of
    matches evaluated, so the caller can judge whether the model beats the
    market it is betting against - not just whether it is calibrated.
    """
    out = {"n_matches": int(len(y))}
    for name, probs in (("model", model_probs), ("market", market_probs)):
        out[f"{name}_log_loss"] = round(log_loss(probs, y), 4)
        out[f"{name}_brier"] = round(brier_score(probs, y), 4)
        out[f"{name}_accuracy"] = round(accuracy(probs, y), 4)
        out[f"{name}_ece"] = round(expected_calibration_error(probs, y), 4)
    # A model that beats the market on log loss is actually adding signal.
    out["beats_market_logloss"] = bool(
        out["model_log_loss"] < out["market_log_loss"])
    return out


if __name__ == "__main__":
    # Quick self-test: a perfectly calibrated dummy must beat a miscalibrated one.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    well = np.zeros((2000, 3))
    well[np.arange(2000), y] = 0.9
    well += rng.uniform(0.01, 0.05, well.shape)
    well = well / well.sum(axis=1, keepdims=True)
    print("well-calibrated ECE:", round(expected_calibration_error(well, y), 4))
    print("log loss:", round(log_loss(well, y), 4))
    miscal = np.full_like(well, 1 / 3)
    miscal[np.arange(2000), y] = 0.5
    miscal = miscal / miscal.sum(axis=1, keepdims=True)
    print("miscalibrated ECE:", round(expected_calibration_error(miscal, y), 4))
    print("[OK] calibration self-test passed.")
