"""
Tests for analysis/loss_attribution.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402
from analysis.loss_attribution import (  # noqa: E402
    attribute_losses, write_loss_report,
)

N_MATCHES = 600


@pytest.fixture(scope="module")
def result():
    data = pipeline.generate_match_data(n_matches=N_MATCHES, seed=42)
    return pipeline.run_backtest(data, use_ml=True, use_rl=True,
                                 save_results=False, verbose=False)


def test_attribute_losses_keys(result):
    a = attribute_losses(result)
    assert a["n_bets"] == len(result["bets_df"])
    for k in ("avg_advertised_edge_pct", "avg_realized_edge_pct",
              "selection_loss_pct", "betting_region_cal_gap_pct",
              "avg_overround_pct", "margin_drag_pct", "avg_clv_pct",
              "clv_t_stat", "model_log_loss", "market_log_loss",
              "dominant_mechanism"):
        assert k in a, f"missing {k}"
    assert a["dominant_mechanism"] in ("margin_drag", "selection_loss",
                                       "calibration_gap", "no_bets")


def test_selection_loss_is_gap_between_advertised_and_realised(result):
    a = attribute_losses(result)
    assert a["selection_loss_pct"] == pytest.approx(
        a["avg_advertised_edge_pct"] - a["avg_realized_edge_pct"], abs=0.02)


def test_report_written_and_readable(result, tmp_path):
    out = tmp_path / "why.txt"
    text = write_loss_report(result, out)
    assert out.exists()
    assert "WHY IS THE MODEL LOSING?" in text
    assert "Bets analysed:" in text
    assert "DOMINANT:" in text
