"""
Tests for models/calibration.py (ECE, Brier, implied probs, isotonic,
model-vs-market comparison).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import calibration  # noqa: E402
from models.calibration import (  # noqa: E402
    accuracy, bookie_probs_matrix, brier_score, compare_to_market,
    expected_calibration_error, implied_probs, isotonic_apply, isotonic_fit,
    log_loss, reliability_curve,
)


def test_log_loss_and_brier_perfect_vs_worst():
    y = np.array([2, 0, 1, 2])
    perfect = np.eye(3)[y]
    assert log_loss(perfect, y) == pytest.approx(0.0, abs=1e-9)
    assert brier_score(perfect, y) == pytest.approx(0.0, abs=1e-9)
    wrong = np.eye(3)[(y + 1) % 3]
    assert log_loss(wrong, y) > 1.0


def test_ece_separates_calibrated_from_miscalibrated():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 4000)
    # well calibrated: probabilities match outcome frequencies per class
    well = np.zeros((4000, 3))
    well[np.arange(4000), y] = 0.85
    well += rng.uniform(0.02, 0.1, well.shape)
    well = well / well.sum(axis=1, keepdims=True)
    miscal = well.copy()
    miscal[:, 2] = np.clip(miscal[:, 2] + 0.25, 0.05, 0.99)  # inflate home win
    miscal = miscal / miscal.sum(axis=1, keepdims=True)
    assert expected_calibration_error(well, y) < \
        expected_calibration_error(miscal, y)


def test_reliability_curve_shapes():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 500)
    probs = rng.dirichlet(np.ones(3), 500)
    curves = reliability_curve(probs, y)
    assert set(curves) == {"home_win", "draw", "away_win"}
    for k, (preds, obs, counts) in curves.items():
        assert len(preds) == len(obs) == len(counts)
        assert np.all((preds >= 0) & (preds <= 1))
        assert np.all((obs >= 0) & (obs <= 1))


def test_implied_probs_remove_overround():
    odds = {"home_win": 2.0, "draw": 3.6, "away_win": 4.0}
    ip = implied_probs(odds)
    assert sum(ip.values()) == pytest.approx(1.0, abs=1e-9)
    # raw implied sum is 1.0/2 + 1.0/3.6 + 1.0/4 = 1.028; normalising by a
    # total > 1 shrinks every probability (the margin is divided out)
    assert ip["home_win"] == pytest.approx((1.0 / 2.0) / 1.02778, abs=1e-3)


def test_bookie_probs_matrix_orders_classes():
    df = __import__("pandas").DataFrame({
        "home": ["A"], "draw": ["B"], "away": ["C"],
        "oh": [2.0], "od": [3.6], "oa": [4.0],
    })
    mat = bookie_probs_matrix(df, "oh", "od", "oa")
    assert mat.shape == (1, 3)
    assert abs(mat[0].sum() - 1.0) < 1e-9
    # [away, draw, home] ordering: home (1/2 implied) must be last
    assert mat[0, 2] == pytest.approx(0.5 / 1.0277, abs=1e-3)


def test_isotonic_apply_renormalises_and_stays_valid():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 600)
    probs = rng.dirichlet(np.ones(3), 600)
    regs = isotonic_fit(probs, y)
    out = isotonic_apply(probs, regs)
    assert out.shape == probs.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all((out >= 0) & (out <= 1))


def test_compare_to_market_reports_all_metrics():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 300)
    model = rng.dirichlet(np.ones(3), 300)
    market = rng.dirichlet(np.ones(3), 300)
    cmp = compare_to_market(model, market, y)
    assert cmp["n_matches"] == 300
    for side in ("model", "market"):
        for metric in ("log_loss", "brier", "accuracy", "ece"):
            assert f"{side}_{metric}" in cmp
    assert isinstance(cmp["beats_market_logloss"], bool)
    # an oracle model beats a random market
    oracle = np.eye(3)[y]
    cmp2 = compare_to_market(oracle, market, y)
    assert cmp2["beats_market_logloss"] is True
    assert cmp2["model_log_loss"] < cmp2["market_log_loss"]


def test_accuracy_consistency():
    y = np.array([0, 1, 2])
    assert accuracy(np.eye(3), y) == 1.0
    assert accuracy(np.roll(np.eye(3), 1, axis=1), y) == 0.0
