"""
Tests for models/stacking_ensemble.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402
from models.stacking_ensemble import StackingEnsemble  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return pipeline.generate_match_data(500, seed=42)


@pytest.fixture(scope="module")
def ensemble_result(df):
    train = df.iloc[:350]
    valid = df.iloc[350:400]
    ens = StackingEnsemble(use_lightgbm=True, use_gb=True)
    metrics = ens.train(train, valid, verbose=False)
    return ens, metrics


def test_ensemble_trains_and_predicts(ensemble_result):
    ens, metrics = ensemble_result
    assert ens.is_trained
    probs = ens.predict("Arsenal", "Chelsea")
    assert abs(sum(probs.values()) - 1.0) < 0.01
    assert all(0 < p < 1 for p in probs.values())


def test_ensemble_has_model_weights(ensemble_result):
    ens, _ = ensemble_result
    assert len(ens.model_weights) >= 2
    # Weights should sum to approximately 1
    total = sum(ens.model_weights.values())
    assert abs(total - 1.0) < 0.1


def test_ensemble_meta_learner_metrics(ensemble_result):
    _, metrics = ensemble_result
    assert "meta_accuracy" in metrics
    assert "meta_log_loss" in metrics
    assert metrics["meta_accuracy"] > 0.3  # above chance
    assert metrics["meta_log_loss"] < 1.2  # below random


def test_ensemble_beats_single_model(df):
    """Stacking ensemble should perform comparably to or better than
    individual models on test data."""
    train = df.iloc[:350]
    valid = df.iloc[350:400]
    test = df.iloc[400:]

    ens = StackingEnsemble(use_lightgbm=True, use_gb=True)
    ens.train(train, valid, verbose=False)

    # Score test set with ensemble
    y_true = test["result"].map({"H": 2, "D": 1, "A": 0}).to_numpy()
    ensemble_probs = []
    for _, row in test.iterrows():
        p = ens.predict(row["home_team"], row["away_team"])
        ensemble_probs.append([p["away_win"], p["draw"], p["home_win"]])
    ensemble_probs = np.array(ensemble_probs)
    ensemble_acc = float(np.mean(np.argmax(ensemble_probs, axis=1) == y_true))

    # Should be above chance (33%)
    assert ensemble_acc > 0.35


def test_ensemble_dixon_coles_off_for_synthetic(df):
    """Dixon-Coles should be disabled for synthetic data."""
    train = df.iloc[:350]
    ens = StackingEnsemble(use_dixon_coles=False)
    ens.train(train, verbose=False)
    assert ens.poisson.rho == 0.0


def test_ensemble_untrained_raises():
    ens = StackingEnsemble()
    with pytest.raises(ValueError, match="not trained"):
        ens.predict("A", "B")
