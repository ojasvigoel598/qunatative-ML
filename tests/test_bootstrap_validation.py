"""
Tests for scripts/16_bootstrap_validation.py (bootstrap, walk-forward, permutation).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bootstrap_validation", ROOT / "scripts" / "16_bootstrap_validation.py")
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)  # noqa: E402


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_metric_basic():
    """Bootstrap CI should contain the point estimate."""
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 2.0, 200)
    result = bv.bootstrap_metric(values, np.mean, n_boot=1000, seed=42)
    assert abs(result["point_estimate"] - 5.0) < 1.0
    assert result["ci_low"] < result["point_estimate"] < result["ci_high"]
    assert result["n_observations"] == 200


def test_bootstrap_roi_ci():
    """Bootstrap ROI CI from synthetic bets."""
    rng = np.random.default_rng(1)
    profits = rng.choice([-10, 15, -8, 20, -5, 12], size=50)
    bets_df = pd.DataFrame({"profit_loss": profits})
    result = bv.bootstrap_roi_ci(bets_df, initial_bankroll=1000.0, n_boot=500, seed=42)
    assert "point_estimate" in result
    assert "ci_low" in result
    assert "ci_high" in result
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


def test_bootstrap_sharpe_ci():
    """Bootstrap Sharpe CI from an equity curve."""
    equity = [1000, 1010, 995, 1020, 1005, 1030, 1015, 1040]
    result = bv.bootstrap_sharpe_ci(equity, n_boot=500, seed=42)
    assert "point_estimate" in result
    assert "ci_low" in result


# ---------------------------------------------------------------- walk-forward
def test_walk_forward_runs():
    """Walk-forward validation should produce per-fold results."""
    df = pipeline.generate_match_data(400, seed=42)
    wf = bv.walk_forward_validation(df, n_seasons=4, verbose=False)
    assert len(wf) >= 2
    for col in ("fold", "accuracy", "log_loss", "roi_pct", "n_bets"):
        assert col in wf.columns
    # Accuracy should be in a plausible range
    assert wf["accuracy"].between(0.3, 0.8).all()


# ---------------------------------------------------------------- permutation
def test_permutation_test_null():
    """Two identical models should give p ≈ 1 (no significant difference)."""
    rng = np.random.default_rng(0)
    profits = rng.choice([-10, 15, -8, 20], size=30)
    df_same = pd.DataFrame({"profit_loss": profits})
    result = bv.paired_permutation_test(df_same, df_same, n_perm=1000, seed=42)
    assert result["observed_difference"] == 0.0
    assert result["p_value"] > 0.8  # null should not be rejected


def test_permutation_test_different():
    """Two very different models should give a small p-value."""
    a = pd.DataFrame({"profit_loss": np.full(30, 10.0)})
    b = pd.DataFrame({"profit_loss": np.full(30, -10.0)})
    result = bv.paired_permutation_test(a, b, n_perm=1000, seed=42)
    assert result["observed_difference"] > 0
    assert result["p_value"] < 0.01  # highly significant
    assert result["significant_at_005"] is True


def test_write_validation_report(tmp_path):
    """Report writer should produce a readable markdown file."""
    wf = pd.DataFrame({
        "fold": [1, 2], "accuracy": [0.55, 0.53],
        "log_loss": [0.95, 0.97], "ece": [0.08, 0.10],
        "roi_pct": [2.1, -1.5], "sharpe_ratio": [0.3, -0.1],
        "sortino_ratio": [0.5, -0.2], "max_drawdown_pct": [5.0, 8.0],
        "n_bets": [15, 12],
    })
    out = tmp_path / "validation_report.md"
    text = bv.write_validation_report(wf, {"ROI": {"point_estimate": 0.3}}, out)
    assert out.exists()
    assert "Walk-Forward" in text
    assert "Bootstrap" in text
