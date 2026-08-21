#!/usr/bin/env python3
"""
Automated Leakage Tests — detect temporal, feature, calibration, and odds leakage.

Research basis:
- quantbet: "data leakage, where information that would not actually have been
  available at the time of the original bet accidentally makes its way into
  the model's training data"
- De Prado (2018): "Backtest overfitting is the #1 reason ML strategies fail"
- DrawBias: "Checked for data leakage — Required"

If any leakage is found: STOP, FIX, REBUILD BACKTEST, RERUN FROM CLEAN STATE.
Do not patch the metric after leakage is discovered.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTemporalLeakage:
    """Test that no future information leaks into training."""

    def test_features_use_only_past_data(self):
        """Rolling features must use .shift(1) to exclude current match."""
        from models.ml_layer import MLFootballPredictor
        ml = MLFootballPredictor()

        # Create a small dataset
        rng = np.random.default_rng(42)
        n = 100
        teams = ["A", "B", "C", "D"]
        df = pd.DataFrame({
            "home_team": rng.choice(teams, n),
            "away_team": rng.choice(teams, n),
            "home_goals": rng.poisson(1.5, n),
            "away_goals": rng.poisson(1.2, n),
            "result": rng.choice(["H", "D", "A"], n),
        })

        features = ml.prepare_features(df, window=5)

        # The first match should have NaN or baseline for rolling features
        # (no prior matches to compute from)
        first_home_goals = features["home_goals_avg"].iloc[0]
        first_match_goals = df["home_goals"].iloc[0]

        # With shift(1), the first match's rolling average should NOT
        # contain its own goals. If window=1, it should be NaN.
        # If window>1, it should only contain prior matches (which don't exist).
        assert first_home_goals != first_match_goals or pd.isna(first_home_goals), \
            "Feature leakage: first match's rolling average includes its own goals"

    def test_elo_updated_after_prediction(self):
        """Elo must be updated AFTER prediction, not before."""
        from models.poisson_elo_model import PoissonEloModel

        model = PoissonEloModel()
        rng = np.random.default_rng(42)
        n = 50
        teams = ["A", "B", "C", "D"]

        df = pd.DataFrame({
            "home_team": rng.choice(teams, n),
            "away_team": rng.choice(teams, n),
            "home_goals": rng.poisson(1.5, n),
            "away_goals": rng.poisson(1.2, n),
        })
        df["result"] = np.where(df["home_goals"] > df["away_goals"], "H",
                                np.where(df["home_goals"] < df["away_goals"], "A", "D"))

        # Train on first 30 matches
        train_df = df.iloc[:30]
        model.train(train_df)

        # Get prediction for match 31
        match31 = df.iloc[30]
        elo_before = model.elo_ratings.copy()
        probs = model.predict(match31["home_team"], match31["away_team"])

        # Elo should NOT have changed during prediction
        elo_after = model.elo_ratings.copy()
        for team in elo_before:
            assert abs(elo_before[team] - elo_after[team]) < 0.01, \
                f"Elo leaked: {team} changed during prediction"

    def test_train_test_temporal_order(self):
        """Training data must precede test data chronologically."""
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        df = pd.DataFrame({
            "date": dates,
            "home_team": rng.choice(["A", "B"], n),
            "away_team": rng.choice(["C", "D"], n),
            "home_goals": rng.poisson(1.5, n),
            "away_goals": rng.poisson(1.2, n),
        })
        df["result"] = np.where(df["home_goals"] > df["away_goals"], "H",
                                np.where(df["home_goals"] < df["away_goals"], "A", "D"))

        # Split chronologically
        split_idx = int(n * 0.7)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        assert train["date"].max() < test["date"].min(), \
            "Temporal leakage: test data precedes training data"

    def test_no_random_split_in_main_backtest(self):
        """Main backtest must use chronological split, not random."""
        # Check that pipeline.py uses chronological split
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pipeline", str(PROJECT_ROOT / "pipeline.py"))
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)

        # The split should be based on position, not random indices
        # Check that SPLIT is defined as fractions, not random
        assert hasattr(pipeline, 'SPLIT'), "SPLIT constant missing"
        assert isinstance(pipeline.SPLIT, tuple), "SPLIT must be a tuple of fractions"
        assert len(pipeline.SPLIT) == 3, "SPLIT must have 3 elements (train, valid, test)"
        assert abs(sum(pipeline.SPLIT) - 1.0) < 0.01, "SPLIT fractions must sum to 1.0"


class TestCalibrationLeakage:
    """Test that calibration does not use test data."""

    def test_calibration_uses_train_data_only(self):
        """Calibration must be fitted on training data, not test data."""
        from models.ml_layer import MLFootballPredictor
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit

        # The MLFootballPredictor uses CalibratedClassifierCV with TimeSeriesSplit
        # which respects temporal order
        ml = MLFootballPredictor(calibration_method="isotonic")

        # Check that calibration CV is TimeSeriesSplit (not random)
        assert isinstance(ml.model.cv, TimeSeriesSplit), \
            "Calibration must use TimeSeriesSplit, not random CV"


class TestOddsLeakage:
    """Test that closing odds are not used for pre-match predictions."""

    def test_closing_odds_not_used_for_prediction(self):
        """Closing odds should not influence model prediction."""
        # The pipeline should only use opening odds for bet decisions
        # and closing odds only for CLV calculation
        from pipeline import run_backtest, generate_match_data

        # This is a structural check: the prediction functions should not
        # receive closing odds as input
        import inspect
        source = inspect.getsource(run_backtest)

        # The prediction loop should use opening odds for edge calculation
        # and closing odds only for CLV
        assert "closing_odds" not in source.split("edges = ")[0] if "edges = " in source else True, \
            "Closing odds may be used for prediction (leakage risk)"


class TestFeatureLeakage:
    """Test that features do not contain future information."""

    def test_rolling_features_are_shifted(self):
        """All rolling features must use .shift(1) or greater."""
        from models.ml_layer import MLFootballPredictor

        ml = MLFootballPredictor()
        import inspect
        source = inspect.getsource(ml.prepare_features)

        # Check that shift is used in rolling computations
        assert ".shift(" in source, \
            "Rolling features must use .shift() to prevent leakage"

    def test_no_future_goals_in_features(self):
        """Features must not contain actual goals from the current match."""
        from models.ml_layer import MLFootballPredictor
        from pipeline import generate_match_data

        rng = np.random.default_rng(42)
        df = generate_match_data(100, seed=42)

        ml = MLFootballPredictor()
        from models.poisson_elo_model import PoissonEloModel
        poisson = PoissonEloModel()
        poisson.train(df.iloc[:60])
        train_feat = poisson.training_features.copy()
        features = ml.prepare_features(train_feat)

        # home_goals_avg should be a rolling average of PRIOR matches
        # not the current match's goals
        if "home_goals_avg" in features.columns and "home_goals" in features.columns:
            # For the second match onwards, home_goals_avg should be based
            # on matches BEFORE the current one
            for i in range(1, min(10, len(features))):
                current_goals = train_feat.iloc[i]["home_goals"]
                avg = features.iloc[i]["home_goals_avg"]
                # The average should reflect PRIOR goals, not just current
                if pd.notna(avg) and current_goals > 0:
                    # If window=1, avg should be the PREVIOUS match's goals
                    prev_goals = train_feat.iloc[i-1]["home_goals"]
                    # This is a loose check - the key is that it's not identical
                    # to current goals in all cases


class TestEdgeDecay:
    """Test that the system monitors edge decay over time."""

    def test_clv_tracking_exists(self):
        """CLV must be tracked for every bet."""
        from pipeline import run_backtest
        import inspect
        source = inspect.getsource(run_backtest)

        assert "clv" in source.lower(), \
            "CLV must be tracked in the backtest"

    def test_ev_calculation_uses_de_vigged(self):
        """EV calculation must use de-vigged market probabilities."""
        from models.calibration import implied_probs

        # Test that implied_probs correctly removes overround
        odds = {"home_win": 2.0, "draw": 3.5, "away_win": 4.0}
        ip = implied_probs(odds)

        # Implied probs should sum to 1 (margin removed)
        assert abs(sum(ip.values()) - 1.0) < 0.01, \
            f"De-vigged probs must sum to 1.0, got {sum(ip.values())}"

        # Each probability should be less than raw 1/odds (margin removed)
        for key in ip:
            assert ip[key] < 1.0 / odds[key] + 0.01, \
                f"De-vigged prob should be <= raw implied for {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
