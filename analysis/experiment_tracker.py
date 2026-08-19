#!/usr/bin/env python3
"""
Experiment Tracker — lightweight JSON-based experiment logging.

Every experiment is logged as a JSON line with:
  - timestamp
  - experiment name / tag
  - configuration (model type, calibration, features, split, seed, etc.)
  - metrics (accuracy, log_loss, ECE, ROI, Sharpe, etc.)
  - optional notes

The log lives at ``backtests/results/experiment_log.jsonl`` and can be
queried, filtered, and compared with the helpers below.

Usage:
    from analysis.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()
    tracker.log(
        name="lightgbm_isotonic_rich_features",
        config={"ml_type": "lightgbm", "calibration": "isotonic",
                "features": "rich", "seed": 42},
        metrics={"accuracy": 0.56, "log_loss": 0.93, "ece": 0.06,
                 "roi_pct": -2.1, "sharpe": -0.15},
        notes="Isotonic calibration with LightGBM on 10-feature set"
    )
    tracker.compare("lightgbm_isotonic_rich_features", "gb_sigmoid_basic")
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "backtests" / "results" / "experiment_log.jsonl"


class ExperimentTracker:
    """Append-only JSONL experiment logger with comparison helpers."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or LOG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- log
    def log(self, name: str, config: Dict[str, Any], metrics: Dict[str, float],
            notes: str = "") -> dict:
        """Append one experiment record and return it."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "config": config,
            "metrics": metrics,
            "notes": notes,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    # ---------------------------------------------------------------- read
    def read_all(self) -> List[dict]:
        """Read every experiment record from the log."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def find(self, name: str) -> List[dict]:
        """Find all records matching an experiment name."""
        return [r for r in self.read_all() if r["name"] == name]

    def latest(self, name: Optional[str] = None) -> Optional[dict]:
        """Return the most recent record, optionally filtered by name."""
        records = self.find(name) if name else self.read_all()
        return records[-1] if records else None

    # ---------------------------------------------------------------- compare
    def compare(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """Compare the latest runs of two experiments.

        Returns a dict with both experiments' configs and metrics, plus
        the differences for every numeric metric.
        """
        a = self.latest(name_a)
        b = self.latest(name_b)
        if a is None or b is None:
            return {"error": f"Missing experiment: "
                    f"{'None' if a is None else name_a}, "
                    f"{'None' if b is None else name_b}"}

        diff = {}
        all_keys = set(a["metrics"]) | set(b["metrics"])
        for k in sorted(all_keys):
            va = a["metrics"].get(k)
            vb = b["metrics"].get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diff[k] = round(va - vb, 6)

        return {
            "experiment_a": {"name": name_a, "config": a["config"],
                             "metrics": a["metrics"]},
            "experiment_b": {"name": name_b, "config": b["config"],
                             "metrics": b["metrics"]},
            "diff_a_minus_b": diff,
        }

    # ---------------------------------------------------------------- display
    def summary_table(self, last_n: int = 10) -> str:
        """Markdown table of the last N experiments."""
        records = self.read_all()[-last_n:]
        if not records:
            return "No experiments logged yet."

        # collect all metric keys
        all_metrics = []
        for r in records:
            for k in r["metrics"]:
                if k not in all_metrics:
                    all_metrics.append(k)

        header = "| Name | " + " | ".join(all_metrics) + " |"
        sep = "|---" * (1 + len(all_metrics)) + "|"
        rows = [header, sep]
        for r in records:
            vals = [f'{r["metrics"].get(k, "")}' for k in all_metrics]
            rows.append(f'| {r["name"]} | ' + " | ".join(vals) + " |")
        return "\n".join(rows)

    def clear(self):
        """Clear the experiment log (use with caution)."""
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    tracker = ExperimentTracker()
    records = tracker.read_all()
    if not records:
        print("No experiments logged yet.")
        print(f"Log file: {tracker.path}")
        sys.exit(0)

    print(f"Found {len(records)} experiments in {tracker.path}\n")
    print(tracker.summary_table())

    if len(records) >= 2:
        a, b = records[-2]["name"], records[-1]["name"]
        print(f"\n--- Comparing last two: {a} vs {b} ---")
        cmp = tracker.compare(a, b)
        if "error" not in cmp:
            for k, v in cmp["diff_a_minus_b"].items():
                sign = "+" if v > 0 else ""
                print(f"  {k}: {sign}{v}")
