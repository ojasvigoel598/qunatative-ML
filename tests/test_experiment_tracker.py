"""
Tests for analysis/experiment_tracker.py.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.experiment_tracker import ExperimentTracker  # noqa: E402


@pytest.fixture
def tracker(tmp_path):
    """Tracker with a temporary log file."""
    return ExperimentTracker(path=tmp_path / "test_log.jsonl")


def test_log_and_read(tracker):
    record = tracker.log("exp_a", {"ml_type": "gb"}, {"accuracy": 0.55})
    assert record["name"] == "exp_a"
    assert record["config"]["ml_type"] == "gb"
    assert record["metrics"]["accuracy"] == 0.55
    assert "timestamp" in record

    all_records = tracker.read_all()
    assert len(all_records) == 1
    assert all_records[0]["name"] == "exp_a"


def test_find_and_latest(tracker):
    tracker.log("exp_a", {}, {"acc": 0.5})
    tracker.log("exp_b", {}, {"acc": 0.6})
    tracker.log("exp_a", {}, {"acc": 0.55})

    found = tracker.find("exp_a")
    assert len(found) == 2

    latest = tracker.latest("exp_a")
    assert latest["metrics"]["acc"] == 0.55

    latest_any = tracker.latest()
    assert latest_any["name"] == "exp_a"
    assert latest_any["metrics"]["acc"] == 0.55


def test_compare(tracker):
    tracker.log("model_a", {"type": "gb"}, {"accuracy": 0.55, "ece": 0.10})
    tracker.log("model_b", {"type": "lgbm"}, {"accuracy": 0.58, "ece": 0.06})

    cmp = tracker.compare("model_a", "model_b")
    assert "error" not in cmp
    assert cmp["diff_a_minus_b"]["accuracy"] == pytest.approx(-0.03, abs=1e-6)
    assert cmp["diff_a_minus_b"]["ece"] == pytest.approx(0.04, abs=1e-6)


def test_compare_missing(tracker):
    cmp = tracker.compare("nonexistent_a", "nonexistent_b")
    assert "error" in cmp


def test_summary_table(tracker):
    tracker.log("exp1", {}, {"acc": 0.5})
    tracker.log("exp2", {}, {"acc": 0.6, "loss": 0.9})
    table = tracker.summary_table()
    assert "exp1" in table
    assert "exp2" in table
    assert "acc" in table
    assert "loss" in table


def test_clear(tracker):
    tracker.log("temp", {}, {"acc": 0.5})
    assert len(tracker.read_all()) == 1
    tracker.clear()
    assert len(tracker.read_all()) == 0


def test_multiple_metrics(tracker):
    tracker.log("full", {"seed": 42, "ml": "lgbm"}, {
        "accuracy": 0.56, "log_loss": 0.93, "ece": 0.06,
        "roi_pct": -2.1, "sharpe": -0.15, "max_drawdown_pct": 12.5,
    })
    record = tracker.latest()
    assert len(record["metrics"]) == 6
    assert record["config"]["seed"] == 42
