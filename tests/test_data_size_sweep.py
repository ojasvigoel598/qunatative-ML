"""
Tests for scripts/15_data_size_sweep.py (fast synthetic subset).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402

# the script filename starts with a digit (15_...), so load it by path
_spec = importlib.util.spec_from_file_location(
    "data_size_sweep", ROOT / "scripts" / "15_data_size_sweep.py")
data_size_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_size_sweep)  # noqa: E402


def test_sweep_runs_and_has_expected_columns():
    df = pipeline.generate_match_data(n_matches=220, seed=42)
    res = data_size_sweep.run_size_sweep(df, sizes=[60, 120],
                                         eval_window=40, n_samples=20,
                                         verbose=False)
    assert len(res) == 2
    for col in ("n_train", "model_log_loss", "market_log_loss", "model_ece",
                "beats_market", "avg_unc_pct", "bet_n_bets", "bet_roi_pct",
                "bet_avg_edge_pct", "bet_realized_edge_pct", "bet_clv_t",
                "bet_adj_n_bets", "bet_adj_roi_pct"):
        assert col in res.columns, f"missing {col}"
    assert (res["n_train"] == [60, 120]).all()
    # small-sample sanity: log loss must be in a plausible band, and the
    # uncertainty must be larger for the smaller training set
    for _, r in res.iterrows():
        assert 0.8 <= r["model_log_loss"] <= 1.25
    assert res.loc[0, "avg_unc_pct"] > res.loc[1, "avg_unc_pct"]


def test_sweep_report_written(tmp_path):
    df = pipeline.generate_match_data(n_matches=220, seed=42)
    res = data_size_sweep.run_size_sweep(df, sizes=[60, 120],
                                         eval_window=40, n_samples=20,
                                         verbose=False)
    out = tmp_path / "report.md"
    text = data_size_sweep.write_report(res, out)
    assert out.exists()
    assert "Data-size sweep" in text
    assert "## Bottleneck diagnosis" in text
    assert "## Uncertainty-adjusted edge filter" in text
