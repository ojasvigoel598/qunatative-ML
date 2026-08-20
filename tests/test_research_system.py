#!/usr/bin/env python3
"""Tests for the autonomous research system components."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ======================================================================
# Frozen Judge tests
# ======================================================================
class TestFrozenJudge:
    """Test the frozen judge evaluation machinery."""

    def test_import(self):
        from evaluation.frozen_judge import FrozenJudge
        judge = FrozenJudge()
        assert judge is not None

    def test_temporal_split(self):
        from evaluation.frozen_judge import FrozenJudge
        import pipeline
        df = pipeline.generate_match_data(300, seed=42)
        judge = FrozenJudge()
        train, valid, embargo, test = judge.temporal_split(df)
        assert len(train) > 0
        assert len(valid) > 0
        assert len(test) > 0
        assert len(embargo) >= 0
        # All splits should be chronological
        assert train["date"].iloc[-1] <= valid["date"].iloc[0]
        assert valid["date"].iloc[-1] <= test["date"].iloc[0]

    def test_gates_pass_on_good_metrics(self):
        from evaluation.frozen_judge import FrozenJudge
        judge = FrozenJudge()
        good_metrics = {
            "total_bets": 50,
            "roi_pct": 5.0,
            "max_drawdown_pct": 15.0,
            "ece": 0.05,
            "sharpe_ratio": 1.5,
            "longest_losing_streak": 5,
            "profit_factor": 1.5,
            "clv_t_stat": 2.5,
            "strike_rate": 45.0,
        }
        gates = judge.evaluate_gates(good_metrics, {"ci_high": 10.0})
        assert all(gates.values()), f"Some gates failed: {gates}"

    def test_gates_fail_on_bad_roi(self):
        from evaluation.frozen_judge import FrozenJudge
        judge = FrozenJudge()
        bad_metrics = {
            "total_bets": 50,
            "roi_pct": -10.0,
            "max_drawdown_pct": 15.0,
            "ece": 0.05,
            "sharpe_ratio": 1.0,
            "longest_losing_streak": 5,
            "profit_factor": 1.2,
            "clv_t_stat": 2.0,
            "strike_rate": 45.0,
        }
        gates = judge.evaluate_gates(bad_metrics, {"ci_high": 5.0})
        assert not gates["min_roi_pct"]

    def test_gates_fail_on_low_bets(self):
        from evaluation.frozen_judge import FrozenJudge
        judge = FrozenJudge()
        metrics = {"total_bets": 5, "roi_pct": 10.0}
        gates = judge.evaluate_gates(metrics)
        assert not gates["min_bets"]

    def test_monte_carlo_gates(self):
        from evaluation.frozen_judge import FrozenJudge
        judge = FrozenJudge()
        mc_good = {"prob_positive_roi": 0.65, "median_roi_pct": 2.0,
                   "prob_ruin": 0.05}
        gates = judge.evaluate_monte_carlo(mc_good)
        assert all(gates.values())

    def test_holdout_lock(self):
        from evaluation.frozen_judge import FrozenJudge
        import pipeline
        df = pipeline.generate_match_data(100, seed=42)
        judge = FrozenJudge()
        h1 = judge.lock_holdout(df)
        assert len(h1) == 16  # SHA-256 truncated to 16 chars
        assert judge.verify_holdout(df, h1)
        # Different data should have different hash
        df2 = pipeline.generate_match_data(100, seed=99)
        assert not judge.verify_holdout(df2, h1)

    def test_transaction_costs(self):
        from evaluation.frozen_judge import FrozenJudge
        judge = FrozenJudge()
        bets = pd.DataFrame({
            "my_odds": [2.0, 2.5],
            "bet_outcome": ["Win", "Lose"],
            "profit_loss": [100, -50],
            "stake": [50, 50],
        })
        adjusted = judge.apply_transaction_costs(bets)
        assert "profit_loss_adjusted" in adjusted.columns
        # Slippage should reduce profits
        assert adjusted["profit_loss_adjusted"].iloc[0] <= bets["profit_loss"].iloc[0]


# ======================================================================
# Monte Carlo Engine tests
# ======================================================================
class TestMonteCarloEngine:
    """Test the Monte Carlo simulation engine."""

    def test_import(self):
        from optimization.monte_carlo_engine import MonteCarloEngine
        engine = MonteCarloEngine()
        assert engine is not None

    def test_basic_simulation(self):
        from optimization.monte_carlo_engine import MonteCarloEngine
        engine = MonteCarloEngine()
        bets = pd.DataFrame({
            "my_odds": [2.0] * 20,
            "bet_outcome": ["Win", "Lose"] * 10,
            "edge_pct": [5.0] * 20,
        })
        result = engine.run(bets, initial_bankroll=1000, n_simulations=1000, seed=42)
        assert result["summary"]["n_simulations"] == 1000
        assert result["summary"]["n_bets"] == 20
        assert 0 <= result["summary"]["prob_positive_roi"] <= 1
        assert 0 <= result["summary"]["prob_ruin"] <= 1

    def test_empty_bets(self):
        from optimization.monte_carlo_engine import MonteCarloEngine
        engine = MonteCarloEngine()
        bets = pd.DataFrame()
        result = engine.run(bets, n_simulations=100, seed=42)
        assert result["summary"]["n_bets"] == 0
        assert result["summary"]["prob_ruin"] == 1.0

    def test_deterministic_with_seed(self):
        from optimization.monte_carlo_engine import MonteCarloEngine
        engine = MonteCarloEngine()
        bets = pd.DataFrame({
            "my_odds": [2.0] * 10,
            "bet_outcome": ["Win"] * 5 + ["Lose"] * 5,
            "edge_pct": [5.0] * 10,
        })
        r1 = engine.run(bets, n_simulations=1000, seed=42)
        r2 = engine.run(bets, n_simulations=1000, seed=42)
        assert r1["summary"]["mean_roi_pct"] == r2["summary"]["mean_roi_pct"]
        assert r1["summary"]["prob_ruin"] == r2["summary"]["prob_ruin"]

    def test_different_seeds_differ(self):
        from optimization.monte_carlo_engine import MonteCarloEngine
        engine = MonteCarloEngine()
        bets = pd.DataFrame({
            "my_odds": [2.0] * 10,
            "bet_outcome": ["Win"] * 5 + ["Lose"] * 5,
            "edge_pct": [5.0] * 10,
        })
        r1 = engine.run(bets, n_simulations=10000, seed=42)
        r2 = engine.run(bets, n_simulations=10000, seed=99)
        # Very unlikely to be exactly equal
        assert r1["summary"]["mean_roi_pct"] != r2["summary"]["mean_roi_pct"]

    def test_higher_kelly_increases_variance(self):
        from optimization.monte_carlo_engine import MonteCarloEngine
        engine = MonteCarloEngine()
        bets = pd.DataFrame({
            "my_odds": [2.0] * 20,
            "bet_outcome": ["Win"] * 10 + ["Lose"] * 10,
            "edge_pct": [5.0] * 20,
        })
        r_low = engine.run(bets, initial_bankroll=1000,
                           n_simulations=5000, kelly_fraction=0.10, seed=42)
        r_high = engine.run(bets, initial_bankroll=1000,
                           n_simulations=5000, kelly_fraction=0.50, seed=42)
        # Higher Kelly should have higher variance
        assert r_high["summary"]["std_roi_pct"] >= r_low["summary"]["std_roi_pct"]


# ======================================================================
# Conformal Prediction tests
# ======================================================================
class TestConformalPrediction:
    """Test the conformal prediction module."""

    def test_import(self):
        from analysis.conformal_prediction import ConformalPredictor
        cp = ConformalPredictor()
        assert cp is not None

    def test_calibration(self):
        from analysis.conformal_prediction import ConformalPredictor
        rng = np.random.default_rng(42)
        n, nc = 200, 3
        y = rng.integers(0, nc, n)
        probs = np.full((n, nc), 0.1)
        probs[np.arange(n), y] = 0.8
        probs += rng.normal(0, 0.05, probs.shape)
        probs = np.clip(probs, 0.01, 0.99)
        probs /= probs.sum(axis=1, keepdims=True)

        cp = ConformalPredictor(confidence_level=0.90)
        threshold = cp.calibrate(probs[:100], y[:100])
        assert 0 <= threshold <= 1

    def test_coverage(self):
        from analysis.conformal_prediction import ConformalPredictor
        rng = np.random.default_rng(42)
        n, nc = 500, 3
        y = rng.integers(0, nc, n)
        probs = np.full((n, nc), 0.1)
        probs[np.arange(n), y] = 0.8
        probs += rng.normal(0, 0.05, probs.shape)
        probs = np.clip(probs, 0.01, 0.99)
        probs /= probs.sum(axis=1, keepdims=True)

        cp = ConformalPredictor(confidence_level=0.90)
        cp.calibrate(probs[:200], y[:200])
        coverage = cp.evaluate_coverage(probs[200:], y[200:])
        # With well-calibrated probs, coverage should be close to nominal
        assert coverage["empirical_coverage"] > 0.75
        assert coverage["empirical_coverage"] < 1.0

    def test_predict_sets(self):
        from analysis.conformal_prediction import ConformalPredictor
        rng = np.random.default_rng(42)
        n, nc = 100, 3
        y = rng.integers(0, nc, n)
        probs = np.full((n, nc), 0.1)
        probs[np.arange(n), y] = 0.8
        probs = np.clip(probs, 0.01, 0.99)
        probs /= probs.sum(axis=1, keepdims=True)

        cp = ConformalPredictor(confidence_level=0.90)
        cp.calibrate(probs[:50], y[:50])
        sets = cp.predict_sets(probs[50:])
        assert len(sets) == 50
        for s in sets:
            assert "prediction_set" in s
            assert 1 <= s["set_size"] <= 3


# ======================================================================
# ROI Attribution tests
# ======================================================================
class TestROIAttribution:
    """Test the ROI attribution system."""

    def test_import(self):
        from analysis.roi_attribution import ROIAttributor
        a = ROIAttributor()
        assert a is not None

    def test_attribution(self):
        from analysis.roi_attribution import ROIAttributor
        a = ROIAttributor()
        baseline = {"roi_pct": -14.3, "total_bets": 26, "sharpe_ratio": -0.48,
                    "sortino_ratio": -0.62, "max_drawdown_pct": 21.3,
                    "avg_edge_pct": 14.8, "avg_clv_pct": 0.01, "clv_t_stat": 0.05,
                    "strike_rate": 42.3, "profit_factor": 0.68}
        new = {"roi_pct": -13.0, "total_bets": 23, "sharpe_ratio": -0.39,
               "sortino_ratio": -0.51, "max_drawdown_pct": 21.8,
               "avg_edge_pct": 14.1, "avg_clv_pct": 0.18, "clv_t_stat": 0.12,
               "strike_rate": 47.8, "profit_factor": 0.70}
        report = a.attribute(baseline, new)
        assert "delta_roi_pct" in report
        assert "contributions" in report
        assert "explanation" in report
        assert abs(report["delta_roi_pct"] - 1.3) < 0.01

    def test_identical_experiments(self):
        from analysis.roi_attribution import ROIAttributor
        a = ROIAttributor()
        metrics = {"roi_pct": -14.3, "total_bets": 26}
        report = a.attribute(metrics, metrics)
        assert abs(report["delta_roi_pct"]) < 0.01


# ======================================================================
# Market Correlation tests
# ======================================================================
class TestMarketCorrelation:
    """Test the market correlation analysis."""

    def test_import(self):
        from analysis.market_correlation import MarketCorrelationAnalyzer
        a = MarketCorrelationAnalyzer()
        assert a is not None

    def test_implied_probabilities(self):
        from analysis.market_correlation import MarketCorrelationAnalyzer
        a = MarketCorrelationAnalyzer()
        odds_h = np.array([2.0, 1.5, 3.0])
        odds_d = np.array([3.5, 4.0, 3.2])
        odds_a = np.array([3.5, 6.0, 2.5])
        probs = a.implied_probabilities(odds_h, odds_d, odds_a)
        assert probs.shape == (3, 3)
        # Each row should sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_analyze(self):
        from analysis.market_correlation import MarketCorrelationAnalyzer
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "result": rng.choice(["H", "D", "A"], n),
            "odds_home_b365": rng.uniform(1.5, 4.0, n),
            "odds_draw_b365": rng.uniform(2.5, 5.0, n),
            "odds_away_b365": rng.uniform(2.0, 6.0, n),
        })
        model_probs = rng.dirichlet(np.ones(3), n)
        a = MarketCorrelationAnalyzer()
        report = a.analyze(df, model_probs)
        assert "correlations" in report
        assert "overround" in report
        assert "head_to_head" in report
        assert report["head_to_head"]["model_accuracy"] > 0


# ======================================================================
# Experiment Registry tests
# ======================================================================
class TestExperimentRegistry:
    """Test the experiment registry."""

    def test_import(self):
        from experiments.experiment_registry import ExperimentRegistry
        r = ExperimentRegistry()
        assert r is not None

    def test_empty_registry(self):
        from experiments.experiment_registry import ExperimentRegistry
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        r = ExperimentRegistry(path=path)
        assert r.read_all() == []
        assert r.summary_stats()["total_experiments"] == 0
        path.unlink()
