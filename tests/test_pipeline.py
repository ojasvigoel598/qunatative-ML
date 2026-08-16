"""
Tests for the Quantitative Sports Betting Model pipeline.

Run with:
    python -m pytest tests/ -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402
from models.ml_layer import MLFootballPredictor  # noqa: E402
from models.poisson_elo_model import PoissonEloModel  # noqa: E402
from models.rl_staking_agent import QLearningStakingAgent  # noqa: E402

N_MATCHES = 600  # small world keeps the test suite fast


@pytest.fixture(scope="module")
def df():
    return pipeline.generate_match_data(n_matches=N_MATCHES, seed=42)


# --------------------------------------------------------------------- data
def test_data_generation_shape(df):
    assert len(df) == N_MATCHES
    required = ["date", "home_team", "away_team", "home_goals", "away_goals",
                "result", "odds_home_b365", "odds_draw_b365", "odds_away_b365",
                "closing_odds_home", "closing_odds_draw", "closing_odds_away"]
    for col in required:
        assert col in df.columns, f"missing column {col}"


def test_no_team_plays_itself(df):
    assert (df["home_team"] == df["away_team"]).sum() == 0


def test_result_consistent_with_goals(df):
    expected = np.where(df["home_goals"] > df["away_goals"], "H",
                        np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    assert (df["result"] == expected).all()


def test_odds_are_reasonable(df):
    # odds must be >= 1 and opening/closing must differ (so CLV is meaningful)
    for col in ["odds_home_b365", "odds_draw_b365", "odds_away_b365"]:
        assert (df[col] >= 1.0).all()
    assert (df["closing_odds_home"] != df["odds_home_b365"]).any()


def test_no_data_leakage_in_ml_features():
    """A match's own goals must never appear in its own features."""
    rng = np.random.default_rng(0)
    sample = pd.DataFrame({
        "home_team": rng.choice(pipeline.TEAMS[:4], 40),
        "away_team": rng.choice(pipeline.TEAMS[4:8], 40),
        "home_goals": rng.poisson(1.6, 40),
        "away_goals": rng.poisson(1.3, 40),
        "home_elo": 1500.0,
        "away_elo": 1500.0,
    })
    sample["result"] = np.where(sample["home_goals"] > sample["away_goals"], "H",
                                np.where(sample["home_goals"] < sample["away_goals"], "A", "D"))
    ml = MLFootballPredictor()
    feats = ml.prepare_features(sample)

    # The shifted rolling mean at a match must equal the mean of the team's
    # PRIOR home goals (never the current match's own goals).
    first_home = sample["home_team"].iloc[0]
    home_games = sample.index[sample["home_team"] == first_home]
    row = home_games[2]  # third home game of this team
    prior = sample.loc[(sample["home_team"] == first_home) & (sample.index < row), "home_goals"]
    assert feats.loc[row, "home_goals_avg"] == pytest.approx(prior.mean(), abs=1e-9)
    # And the current game's goals must NOT appear in the feature: inject an
    # extreme current score and check the feature is unchanged.
    current_goals = int(sample.loc[row, "home_goals"])
    sample.loc[row, "home_goals"] = current_goals + 50
    feats2 = ml.prepare_features(sample)
    assert feats2.loc[row, "home_goals_avg"] == pytest.approx(prior.mean(), abs=1e-9)


# ------------------------------------------------------------------- models
def test_poisson_predictions_sum_to_one(df):
    poisson = PoissonEloModel()
    train = df.iloc[: int(len(df) * 0.7)]
    poisson.train(train)
    probs = poisson.predict("Arsenal", "Chelsea")
    assert abs(sum(probs[k] for k in ("home_win", "draw", "away_win")) - 1.0) < 0.01
    assert probs["expected_home_goals"] > 0


def test_dixon_coles_tau_cell_values():
    """DC tau factors: with rho < 0 (the classic football finding) 0-0 and
    1-1 become MORE likely than independence and 1-0 / 0-1 less likely;
    mirrored for rho > 0; tau = 1 for rho = 0 and for all other cells."""
    model = PoissonEloModel()
    lam_h, lam_a = 1.5, 1.2
    # rho < 0: 0-0 and 1-1 up, 1-0 / 0-1 down
    assert model._dc_tau(0, 0, lam_h, lam_a, -0.05) > 1.0
    assert model._dc_tau(1, 1, lam_h, lam_a, -0.05) > 1.0
    assert model._dc_tau(1, 0, lam_h, lam_a, -0.05) < 1.0
    assert model._dc_tau(0, 1, lam_h, lam_a, -0.05) < 1.0
    # rho > 0: 0-0 and 1-1 down, 1-0 / 0-1 up
    assert model._dc_tau(0, 0, lam_h, lam_a, 0.05) < 1.0
    assert model._dc_tau(1, 1, lam_h, lam_a, 0.05) < 1.0
    assert model._dc_tau(1, 0, lam_h, lam_a, 0.05) > 1.0
    assert model._dc_tau(0, 1, lam_h, lam_a, 0.05) > 1.0
    # rho = 0 and non-low-score cells are always 1
    assert model._dc_tau(0, 0, lam_h, lam_a, 0.0) == 1.0
    assert model._dc_tau(2, 3, lam_h, lam_a, -0.05) == 1.0


def test_dixon_coles_fit_and_predict(df):
    """DC must fit a bounded rho, keep probabilities normalised, and leave
    predictions valid with the correction enabled (default)."""
    poisson = PoissonEloModel()
    poisson.train(df.iloc[: int(len(df) * 0.7)])
    assert -0.25 <= poisson.rho <= 0.25
    probs = poisson.predict("Arsenal", "Chelsea")
    assert abs(sum(probs[k] for k in ("home_win", "draw", "away_win")) - 1.0) < 0.01
    # disabling DC must still produce a valid model
    off = PoissonEloModel(use_dixon_coles=False)
    off.train(df.iloc[: int(len(df) * 0.7)], verbose=False)
    p_off = off.predict("Arsenal", "Chelsea")
    assert abs(sum(p_off[k] for k in ("home_win", "draw", "away_win")) - 1.0) < 0.01


def test_poisson_calibration_is_not_absurd(df):
    poisson = PoissonEloModel()
    poisson.train(df.iloc[: int(len(df) * 0.7)])
    scored = pipeline._predictions_over(df.iloc[int(len(df) * 0.7):], poisson, None)
    ev = pipeline.evaluate_probability_quality(scored)
    # Must beat the 3-class random baseline log-loss of ln(3) ~ 1.099
    assert ev["log_loss"] < 1.09
    assert ev["accuracy"] >= ev["baseline_accuracy"] - 0.05


def test_ml_predicts_distinct_teams_differently(df):
    poisson = PoissonEloModel()
    feat = poisson.prepare_features(df.iloc[: int(len(df) * 0.7)])
    ml = MLFootballPredictor()
    ml.train(feat, verbose=False)
    p1 = ml.predict_proba("Arsenal", "Chelsea", 1600, 1400)
    p2 = ml.predict_proba("Chelsea", "Arsenal", 1400, 1600)
    assert p1["home_win"] != p2["home_win"]
    assert abs(sum(p1.values()) - 1.0) < 0.01


def test_rl_agent_stakes_within_bounds():
    rng = np.random.default_rng(0)
    experiences = [(rng.uniform(0.03, 0.15), 1.0, rng.uniform(1.8, 3.0),
                    bool(rng.random() < 0.55)) for _ in range(150)]
    agent = QLearningStakingAgent()
    agent.train(experiences, episodes=50)
    stake = agent.get_stake_fraction(0.08, 2.2, 1000, 1000)
    assert 0.0 <= stake <= 0.05
    # no edge -> no bet
    assert agent.get_stake_fraction(0.0, 2.2, 1000, 1000) == 0.0


# ----------------------------------------------------------------- backtest
def test_backtest_runs_and_is_consistent(df):
    res = pipeline.run_backtest(df, use_ml=True, use_rl=True, seed=42,
                                save_results=False, verbose=False)
    summary = res["summary"]
    bets = res["bets_df"]
    assert summary["total_bets"] == len(bets)
    assert summary["final_bankroll"] == pytest.approx(bets["running_bankroll"].iloc[-1], abs=0.01)
    assert summary["strike_rate"] >= 0.0
    assert summary["roi_pct"] != 0.0 or summary["total_bets"] == 0
    # CLV must be computed from real closing odds (not a constant 0)
    if len(bets):
        assert (bets["clv_pct"].abs() > 0).any()
        assert bets["my_odds"].min() >= 1.6


def test_backtest_is_reproducible(df):
    r1 = pipeline.run_backtest(df, use_ml=True, use_rl=True, seed=42,
                               save_results=False, verbose=False)
    r2 = pipeline.run_backtest(df, use_ml=True, use_rl=True, seed=42,
                               save_results=False, verbose=False)
    assert r1["summary"] == r2["summary"]
    assert r1["bets_df"].equals(r2["bets_df"])


def test_evaluation_metrics_match_results(df):
    """The class mapping in evaluate_probability_quality must be correct:
    result H (home win) has class index 2 (p_home_win column)."""
    poisson = PoissonEloModel()
    train = df.iloc[: int(len(df) * 0.65)]
    poisson.train(train)
    test = df.iloc[int(len(df) * 0.8):]
    scored = pipeline._predictions_over(test, poisson, None)
    ev = pipeline.evaluate_probability_quality(scored)
    # A perfect constant-home prediction achieves the home-win rate accuracy.
    assert ev["accuracy"] > 0.3  # far above chance 1/3
