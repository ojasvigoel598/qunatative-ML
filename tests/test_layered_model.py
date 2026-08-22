#!/usr/bin/env python3
"""Unit tests for layered model components.

Following the testing skill: write tests first, verify each component independently.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from models.layered_model import (
    BayesianTeamPrior,
    EWMARecency,
    ContextualLayer,
    AdaptiveEnsemble,
    LayeredModel,
)


# ======================================================================
# BayesianTeamPrior Tests
# ======================================================================
class TestBayesianTeamPrior:
    def test_update_increments_count(self):
        bp = BayesianTeamPrior()
        bp.update("TeamA", 2.0, 1.0)
        assert bp.team_stats["TeamA"]["n"] == 1

    def test_multiple_updates(self):
        bp = BayesianTeamPrior()
        bp.update("TeamA", 2.0, 1.0)
        bp.update("TeamA", 3.0, 0.0)
        assert bp.team_stats["TeamA"]["n"] == 2
        assert bp.team_stats["TeamA"]["goals_for"] == 5.0

    def test_strength_unknown_team(self):
        bp = BayesianTeamPrior()
        s = bp.get_strength("Unknown")
        assert s["n_matches"] == 0
        assert s["attack"] == bp.league_mu

    def test_shrinkage_towards_mean(self):
        bp = BayesianTeamPrior(league_mu=1.5)
        bp.update("TeamA", 5.0, 0.0)  # very strong team
        s = bp.get_strength("TeamA")
        # With 1 match, strong shrinkage toward 1.5
        assert s["attack"] < 5.0
        assert s["attack"] > 1.5

    def test_less_shrinkage_with_more_data(self):
        bp = BayesianTeamPrior(league_mu=1.5)
        for _ in range(100):
            bp.update("TeamA", 3.0, 1.0)
        s = bp.get_strength("TeamA")
        # With 100 matches, should be close to raw average (3.0)
        assert abs(s["attack"] - 3.0) < 0.5

    def test_uncertainty_decreases(self):
        bp = BayesianTeamPrior()
        bp.update("TeamA", 2.0, 1.0)
        s1 = bp.get_strength("TeamA")
        bp.update("TeamA", 2.0, 1.0)
        s2 = bp.get_strength("TeamA")
        assert s2["uncertainty"] < s1["uncertainty"]


# ======================================================================
# EWMARecency Tests
# ======================================================================
class TestEWMARecency:
    def test_first_observation(self):
        ewma = EWMARecency()
        ewma.update("TeamA", 2.0)
        assert ewma.get_ewma("TeamA") == 2.0

    def test_subsequent_observations(self):
        ewma = EWMARecency(alpha=0.3)
        ewma.update("TeamA", 2.0)
        ewma.update("TeamA", 3.0)
        # EWMA = 0.3 * 3.0 + 0.7 * 2.0 = 2.3
        assert abs(ewma.get_ewma("TeamA") - 2.3) < 0.01

    def test_unknown_team_returns_default(self):
        ewma = EWMARecency()
        assert ewma.get_ewma("Unknown") == 1.5

    def test_alpha_affects_weight(self):
        ewma_fast = EWMARecency(alpha=0.9)
        ewma_slow = EWMARecency(alpha=0.1)
        for ewma in [ewma_fast, ewma_slow]:
            ewma.update("TeamA", 1.0)
            ewma.update("TeamA", 5.0)
        # Fast alpha should be closer to 5.0
        assert ewma_fast.get_ewma("TeamA") > ewma_slow.get_ewma("TeamA")


# ======================================================================
# ContextualLayer Tests
# ======================================================================
class TestContextualLayer:
    def test_no_data_returns_one(self):
        ctx = ContextualLayer()
        adj = ctx.get_adjustment("Unknown", 1.0, is_home=True)
        assert adj == 1.0

    def test_tired_team_penalty(self):
        ctx = ContextualLayer()
        for _ in range(5):
            ctx.update("TeamA", 1.0, 1.0, is_home=True, rest_days=3)
        adj = ctx.get_adjustment("TeamA", 1.0, is_home=True)
        assert adj < 1.0  # tired team

    def test_well_rested_bonus(self):
        ctx = ContextualLayer()
        for _ in range(5):
            ctx.update("TeamA", 1.0, 1.0, is_home=True, rest_days=10)
        adj = ctx.get_adjustment("TeamA", 1.0, is_home=True)
        assert adj > 1.0  # well-rested

    def test_keeps_only_recent_matches(self):
        ctx = ContextualLayer()
        for i in range(25):
            ctx.update("TeamA", float(i), 1.0, is_home=True)
        assert len(ctx.team_context["TeamA"]["recent_scores"]) == 20


# ======================================================================
# AdaptiveEnsemble Tests
# ======================================================================
class TestAdaptiveEnsemble:
    def test_default_weights_sum_to_one(self):
        ens = AdaptiveEnsemble()
        total = sum(ens.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_better_model_gets_higher_weight(self):
        ens = AdaptiveEnsemble()
        for _ in range(10):
            ens.record_performance("poisson", 0.95)
            ens.record_performance("kde", 0.80)
        ens.update_weights()
        assert ens.weights["kde"] > ens.weights["poisson"]

    def test_combine_normalizes(self):
        ens = AdaptiveEnsemble()
        preds = {
            "poisson": {"home_win": 0.4, "draw": 0.3, "away_win": 0.3},
            "kde": {"home_win": 0.5, "draw": 0.2, "away_win": 0.3},
        }
        combined = ens.combine(preds)
        total = sum(combined.values())
        assert abs(total - 1.0) < 0.01


# ======================================================================
# LayeredModel Integration Test
# ======================================================================
class TestLayeredModel:
    def test_import(self):
        from models.layered_model import LayeredModel
        assert LayeredModel is not None

    def test_predict_requires_training(self):
        model = LayeredModel()
        with pytest.raises(ValueError):
            model.predict("TeamA", "TeamB")
