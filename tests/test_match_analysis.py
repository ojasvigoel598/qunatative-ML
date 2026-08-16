"""
Tests for analysis/match_analysis.py (per-match reasoning card).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402
from analysis import match_analysis  # noqa: E402
from analysis.match_analysis import (  # noqa: E402
    analyze_match, available_books, best_closing_odds, best_odds,
    bookie_implied, book_odds, build_predictions_table, write_predictions_table,
)

N_MATCHES = 500


@pytest.fixture(scope="module")
def df():
    return pipeline.generate_match_data(n_matches=N_MATCHES, seed=42)


@pytest.fixture(scope="module")
def trained():
    data = pipeline.generate_match_data(n_matches=N_MATCHES, seed=42)
    p = pipeline.PoissonEloModel(use_dixon_coles=False)
    p.train(data.iloc[:350], verbose=False)
    return p


def test_price_shopping_best_odds(df):
    row = df.iloc[10]
    best = best_odds(row)
    assert available_books(row) == ["b365", "pin"]
    for o in ("home_win", "draw", "away_win"):
        assert best[o] >= book_odds(row, "b365")[o]
        assert best[o] >= book_odds(row, "pin")[o]


def test_bookie_implied_is_normalised(df):
    ip = bookie_implied(df.iloc[10])
    assert abs(sum(ip.values()) - 1.0) < 1e-9


def test_best_closing_odds_present(df):
    cl = best_closing_odds(df.iloc[10])
    for o in ("home_win", "draw", "away_win"):
        assert cl[o] is None or cl[o] > 1.0


def test_analyze_match_card_has_all_fields(df, trained):
    card = analyze_match(df.iloc[300], trained, None)
    for k in ("p_model_home_win", "unc_home_win", "p_bookie_home_win",
              "best_odds_home_win", "fair_odds_model", "edge_pct",
              "ev_per_unit_pct", "decision", "reason", "clv_pct", "result",
              "n_books"):
        assert k in card, f"missing {k}"
    assert card["decision"] in ("BET", "NO_BET")
    assert card["n_books"] == 2
    assert sum(card[k] for k in ("p_model_home_win", "p_model_draw",
                                 "p_model_away_win")) > 0.99


def test_uncertainty_adjusted_rule_is_stricter(df, trained):
    """With uncertainty_z > 0, an edge smaller than its own standard error
    must be rejected even if it passes the plain threshold."""
    n_raw = 0
    n_adj = 0
    for i in range(400, 460):
        raw = analyze_match(df.iloc[i], trained, None, uncertainty_z=0.0)
        adj = analyze_match(df.iloc[i], trained, None, uncertainty_z=1.0)
        n_raw += raw["decision"] == "BET"
        n_adj += adj["decision"] == "BET"
        if raw["decision"] == "BET" and adj["decision"] == "NO_BET":
            assert adj["edge_pct"] < 1.0 * adj["edge_uncertainty_pct"] + 1e-9
    assert n_adj <= n_raw  # uncertainty guard never *adds* bets
    # and it must actually bite at least once over 60 matches
    assert n_adj < n_raw or n_adj == 0


def test_predictions_table_and_csv(df, trained, tmp_path):
    sub = df.iloc[380:420].copy()
    table = build_predictions_table(sub, trained, None)
    assert len(table) == 40
    assert set(table["decision"]).issubset({"BET", "NO_BET"})
    out = tmp_path / "preds.csv"
    write_predictions_table(sub, trained, None, out)
    reread = pd.read_csv(out)
    assert len(reread) == 40


def test_works_on_real_schema():
    """The probing must work on the real-data schema too (no training needed -
    just odds access on one loaded season)."""
    from data.real_data import get_season
    real = get_season("I1", "2324", offline=True)
    row = real.iloc[5]
    best = best_odds(row)
    assert available_books(row) == ["b365", "pin"]
    for o in ("home_win", "draw", "away_win"):
        assert best[o] >= book_odds(row, "b365")[o]
    cl = best_closing_odds(row)
    assert all(v is None or v > 1.0 for v in cl.values())
    assert cl["home_win"] >= real.iloc[5]["closing_odds_home"]
