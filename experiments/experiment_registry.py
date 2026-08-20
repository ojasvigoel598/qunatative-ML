#!/usr/bin/env python3
"""
Experiment Registry — manages the experiment graph and provides
querying/analysis capabilities.

The registry stores experiments as JSONL and maintains a dependency
graph showing parent-child relationships between experiments.

Usage:
    from experiments.experiment_registry import ExperimentRegistry
    registry = ExperimentRegistry()
    registry.list_experiments()
    registry.get_experiment("E0001")
    registry.print_graph()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REGISTRY_PATH = PROJECT_ROOT / "experiments" / "experiment_registry.jsonl"
GRAPH_PATH = PROJECT_ROOT / "experiments" / "experiment_graph.json"


class ExperimentRegistry:
    """Query and manage the experiment registry."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or REGISTRY_PATH
        self.graph_path = GRAPH_PATH

    def read_all(self) -> List[Dict]:
        """Read all experiments from the registry."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get a specific experiment by ID."""
        for exp in self.read_all():
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None

    def get_accepted(self) -> List[Dict]:
        """Get all accepted experiments."""
        return [e for e in self.read_all() if e.get("decision") == "ACCEPT"]

    def get_rejected(self) -> List[Dict]:
        """Get all rejected experiments."""
        return [e for e in self.read_all() if e.get("decision") == "REJECT"]

    def get_children(self, experiment_id: str) -> List[Dict]:
        """Get all child experiments of a given experiment."""
        children = []
        for exp in self.read_all():
            if exp.get("parent_experiment") == experiment_id:
                children.append(exp)
        return children

    def get_lineage(self, experiment_id: str) -> List[Dict]:
        """Get the full lineage (ancestors) of an experiment."""
        lineage = []
        current = self.get_experiment(experiment_id)
        while current:
            lineage.append(current)
            parent_id = current.get("parent_experiment")
            if parent_id:
                current = self.get_experiment(parent_id)
            else:
                break
        return lineage

    def summary_stats(self) -> Dict:
        """Get summary statistics of all experiments."""
        all_exp = self.read_all()
        accepted = [e for e in all_exp if e.get("decision") == "ACCEPT"]
        rejected = [e for e in all_exp if e.get("decision") == "REJECT"]

        rois = [e.get("summary", {}).get("roi_pct", None)
                for e in all_exp if e.get("summary")]
        rois = [r for r in rois if r is not None]

        return {
            "total_experiments": len(all_exp),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "pending": len(all_exp) - len(accepted) - len(rejected),
            "best_roi": max(rois) if rois else None,
            "worst_roi": min(rois) if rois else None,
            "median_roi": sorted(rois)[len(rois)//2] if rois else None,
        }

    def print_graph(self):
        """Print the experiment dependency graph as ASCII art."""
        if not self.graph_path.exists():
            print("No experiment graph found.")
            return

        with open(self.graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        roots = graph.get("roots", [])

        def _print_tree(node_id: str, prefix: str = "", is_last: bool = True):
            connector = "`-- " if is_last else "|-- "
            exp = self.get_experiment(node_id)
            hypothesis = exp.get("hypothesis", "?")[:50] if exp else "?"
            decision = exp.get("decision", "?") if exp else "?"
            print(f"{prefix}{connector}{node_id} [{decision}] {hypothesis}")
            children = graph.get(node_id, [])
            new_prefix = prefix + ("    " if is_last else "|   ")
            for i, child in enumerate(children):
                _print_tree(child, new_prefix, i == len(children) - 1)

        print("Experiment Graph:")
        for i, root in enumerate(roots):
            _print_tree(root, "", i == len(roots) - 1)

    def print_table(self):
        """Print a formatted table of all experiments."""
        experiments = self.read_all()
        if not experiments:
            print("No experiments recorded.")
            return

        print(f"\n{'ID':<8} {'Decision':<8} {'ROI%':>8} {'Bets':>5} "
              f"{'Sharpe':>8} {'P(>0)':>8} Hypothesis")
        print("-" * 80)
        for exp in experiments:
            s = exp.get("summary", {})
            mc = exp.get("mc_summary", {})
            roi = s.get("roi_pct", "N/A")
            roi_str = f"{roi:+.1f}" if isinstance(roi, (int, float)) else str(roi)
            bets = s.get("total_bets", exp.get("n_bets", "N/A"))
            sharpe = s.get("sharpe_ratio", "N/A")
            sharpe_str = f"{sharpe:.3f}" if isinstance(sharpe, (int, float)) else str(sharpe)
            prob = mc.get("prob_positive_roi", "N/A") if mc else "N/A"
            prob_str = f"{prob:.1%}" if isinstance(prob, (int, float)) else str(prob)
            hyp = exp.get("hypothesis", "")[:45]
            print(f"{exp.get('experiment_id', '?'):<8} "
                  f"{exp.get('decision', '?'):<8} "
                  f"{roi_str:>8} {str(bets):>5} "
                  f"{sharpe_str:>8} {prob_str:>8} {hyp}")


if __name__ == "__main__":
    registry = ExperimentRegistry()
    stats = registry.summary_stats()
    print("Experiment Registry Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()
    registry.print_table()
    print()
    registry.print_graph()
