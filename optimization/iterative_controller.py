#!/usr/bin/env python3
"""
Iterative Controller — the self-improvement orchestration loop.

Implements the research loop:
  Research -> Audit -> Hypothesis -> Experiment -> Test -> Simulate
  -> Evaluate -> Retain/Reject -> Record -> Repeat

The controller maintains structured experiment state and enforces:
1. One change at a time (causal attribution)
2. Frozen judge evaluation (optimizer cannot modify the judge)
3. 1M Monte Carlo validation
4. 10-consecutive-window stability gate
5. Complete experiment tracking

Usage:
    from optimization.iterative_controller import IterativeController
    controller = IterativeController()
    result = controller.run_iteration(
        hypothesis="Isotonic calibration improves ECE",
        change_fn=my_change_fn,
        config=my_config,
    )
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pipeline
from evaluation.frozen_judge import FrozenJudge
from optimization.monte_carlo_engine import MonteCarloEngine
from analysis.roi_attribution import ROIAttributor

EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
EXPERIMENT_REGISTRY = EXPERIMENT_DIR / "experiment_registry.jsonl"
EXPERIMENT_GRAPH = EXPERIMENT_DIR / "experiment_graph.json"


class IterativeController:
    """Orchestrates the iterative research loop.

    Maintains experiment state, enforces the frozen judge, runs Monte Carlo,
    and tracks the full experiment graph.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.judge = FrozenJudge(verbose=verbose)
        self.mc_engine = MonteCarloEngine(verbose=False)
        self.attributor = ROIAttributor(verbose=False)
        self.experiments: List[Dict] = []
        self._load_registry()

    def _load_registry(self):
        """Load existing experiments from the registry."""
        if EXPERIMENT_REGISTRY.exists():
            with open(EXPERIMENT_REGISTRY, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.experiments.append(json.loads(line))
        self.experiment_count = len(self.experiments)
        self._next_id = self.experiment_count + 1

    def _next_experiment_id(self) -> str:
        """Generate next experiment ID."""
        eid = f"E{self._next_id:04d}"
        self._next_id += 1
        return eid

    def _record_experiment(self, experiment: Dict):
        """Append experiment to the registry."""
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        with open(EXPERIMENT_REGISTRY, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment, default=str) + "\n")
        self.experiments.append(experiment)
        self._update_graph()

    def _update_graph(self):
        """Update the experiment dependency graph."""
        graph = {}
        for exp in self.experiments:
            eid = exp["experiment_id"]
            parent = exp.get("parent_experiment")
            if parent:
                if parent not in graph:
                    graph[parent] = []
                graph[parent].append(eid)
            else:
                if "roots" not in graph:
                    graph["roots"] = []
                graph["roots"].append(eid)

        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        with open(EXPERIMENT_GRAPH, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

    def get_baseline(self) -> Optional[Dict]:
        """Get the latest accepted experiment as the baseline."""
        accepted = [e for e in self.experiments
                    if e.get("decision") == "ACCEPT"]
        return accepted[-1] if accepted else None

    # ==================================================================
    # Core iteration
    # ==================================================================
    def run_iteration(
        self,
        hypothesis: str,
        change_fn: Callable[[pd.DataFrame], Dict[str, Any]],
        research_basis: str = "",
        parent_experiment: Optional[str] = None,
        n_simulations: int = 1_000_000,
        seed: int = 42,
    ) -> Dict:
        """Run one iteration of the research loop.

        Args:
            hypothesis: what this experiment tests.
            change_fn: function(df) -> dict that trains models and returns
                summary, bets_df, equity, test_eval, models.
            research_basis: paper/source motivating this hypothesis.
            parent_experiment: experiment_id of the parent (baseline).
            n_simulations: Monte Carlo simulation count.
            seed: random seed.

        Returns:
            Complete experiment record with all results.
        """
        experiment_id = self._next_experiment_id()
        t0 = time.time()

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {experiment_id}: {hypothesis}")
            print(f"{'='*70}")

        # 1. Generate data
        df = pipeline.generate_match_data(1200, seed=seed)

        # 2. Temporal split (frozen judge)
        train, valid, embargo, test = self.judge.temporal_split(df)

        # 3. Lock holdout
        holdout_hash = self.judge.lock_holdout(test)

        # 4. Apply the change and get results
        try:
            change_result = change_fn(df)
        except Exception as e:
            return self._record_failure(experiment_id, hypothesis,
                                        research_basis, parent_experiment,
                                        str(e), time.time() - t0)

        summary = change_result.get("summary", {})
        bets_df = change_result.get("bets_df", pd.DataFrame())
        test_eval = change_result.get("test_eval", {})

        # 5. Apply transaction costs
        if len(bets_df) > 0:
            bets_with_costs = self.judge.apply_transaction_costs(bets_df)
            # Recompute ROI after costs
            if "profit_loss_adjusted" in bets_with_costs.columns:
                adjusted_profit = bets_with_costs["profit_loss_adjusted"].sum()
                summary["roi_pct_adjusted"] = round(
                    adjusted_profit / pipeline.INITIAL_BANKROLL * 100, 2)
        else:
            bets_with_costs = bets_df

        # 6. Bootstrap CI for ROI
        roi_ci = None
        if len(bets_df) > 10:
            from scripts.s16_bootstrap_validation import bootstrap_roi_ci
            try:
                roi_ci = bootstrap_roi_ci(bets_df, pipeline.INITIAL_BANKROLL,
                                          n_boot=2000, seed=seed)
            except Exception:
                pass

        # 7. Frozen judge evaluation
        gate_result = self.judge.evaluate(
            metrics=summary,
            roi_ci=roi_ci,
        )

        # 8. Monte Carlo simulation (only if gates pass)
        mc_results = None
        if gate_result["all_gates_pass"] and len(bets_df) > 5:
            mc_results = self.mc_engine.run(
                bets_df=bets_df,
                initial_bankroll=pipeline.INITIAL_BANKROLL,
                n_simulations=n_simulations,
                seed=seed,
            )
            mc_gate = self.judge.evaluate_monte_carlo(mc_results["summary"])
            gate_result["mc_gate_results"] = mc_gate
            gate_result["all_mc_pass"] = all(mc_gate.values())
            gate_result["passed"] = (gate_result["all_gates_pass"] and
                                     gate_result["all_mc_pass"])

        # 9. ROI attribution
        baseline = self.get_baseline()
        attribution = None
        if baseline and baseline.get("summary"):
            attribution = self.attributor.attribute(
                baseline["summary"], summary,
                baseline.get("test_eval"), test_eval)

        # 10. Decision
        decision = "ACCEPT" if gate_result["passed"] else "REJECT"
        elapsed = time.time() - t0

        # 11. Build experiment record
        experiment = {
            "experiment_id": experiment_id,
            "parent_experiment": parent_experiment or (baseline["experiment_id"] if baseline else None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis": hypothesis,
            "research_basis": research_basis,
            "decision": decision,
            "holdout_hash": holdout_hash,
            "seed": seed,
            "summary": summary,
            "test_eval": test_eval,
            "roi_ci": roi_ci,
            "gate_results": gate_result.get("gate_results", {}),
            "mc_summary": mc_results["summary"] if mc_results else None,
            "mc_gate_results": gate_result.get("mc_gate_results", {}),
            "attribution": attribution,
            "n_bets": len(bets_df),
            "elapsed_seconds": round(elapsed, 1),
            "reason": gate_result.get("summary", ""),
        }

        # 12. Record
        self._record_experiment(experiment)

        # 13. Log to experiment tracker
        try:
            from analysis.experiment_tracker import ExperimentTracker
            tracker = ExperimentTracker()
            tracker.log(
                name=f"{experiment_id}: {hypothesis[:50]}",
                config={"experiment_id": experiment_id,
                        "parent": experiment.get("parent_experiment")},
                metrics=summary,
                notes=f"Decision: {decision}. {gate_result.get('summary', '')}"
            )
        except Exception:
            pass

        if self.verbose:
            print(f"\n  Decision: {decision}")
            if attribution:
                print(f"  Attribution: {attribution.get('explanation', 'N/A')}")
            print(f"  Elapsed: {elapsed:.1f}s")

        return experiment

    def _record_failure(self, experiment_id: str, hypothesis: str,
                        research_basis: str, parent_experiment: Optional[str],
                        error: str, elapsed: float) -> Dict:
        """Record a failed experiment."""
        experiment = {
            "experiment_id": experiment_id,
            "parent_experiment": parent_experiment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis": hypothesis,
            "research_basis": research_basis,
            "decision": "REJECT",
            "error": error,
            "elapsed_seconds": round(elapsed, 1),
            "reason": f"Experiment failed: {error}",
        }
        self._record_experiment(experiment)
        if self.verbose:
            print(f"  FAILED: {error}")
        return experiment

    # ==================================================================
    # Walk-forward 10-window stability
    # ==================================================================
    def run_stability_check(
        self,
        change_fn: Callable[[pd.DataFrame, int], Dict[str, Any]],
        n_windows: int = 10,
        n_simulations: int = 100_000,
        seed: int = 42,
    ) -> Dict:
        """Run 10 independent temporal windows to check stability.

        Args:
            change_fn: function(df, window_id) -> dict with summary, bets_df.
            n_windows: number of independent windows.
            n_simulations: MC simulations per window.
            seed: base RNG seed.

        Returns:
            Stability check results.
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STABILITY CHECK: {n_windows} independent windows")
            print(f"{'='*70}")

        df = pipeline.generate_match_data(2000, seed=seed)
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        window_size = n // n_windows

        window_results = []
        for w in range(n_windows):
            start = w * window_size
            end = min(start + window_size, n)
            window_df = df.iloc[start:end].copy()

            if len(window_df) < 30:
                continue

            try:
                result = change_fn(window_df, w)
                window_results.append({
                    "window_id": w,
                    "metrics": result.get("summary", {}),
                    "n_bets": result.get("summary", {}).get("total_bets", 0),
                })
            except Exception as e:
                window_results.append({
                    "window_id": w,
                    "metrics": {"total_bets": 0, "roi_pct": -999},
                    "error": str(e),
                })

        # Evaluate stability
        stability = self.judge.evaluate_stability(window_results)

        if self.verbose:
            print(f"\n  Consecutive passes: "
                  f"{stability['consecutive_count']}/{stability['required']}")
            print(f"  Overall: {'PASSED' if stability['passed'] else 'FAILED'}")
            for wr in window_results:
                status = "PASS" if wr.get("metrics", {}).get("roi_pct", -999) >= 0 else "FAIL"
                print(f"    Window {wr['window_id']}: {status} "
                      f"(ROI={wr['metrics'].get('roi_pct', 0):+.1f}%, "
                      f"bets={wr.get('n_bets', 0)})")

        return stability

    # ==================================================================
    # Summary
    # ==================================================================
    def summary_table(self) -> str:
        """Generate a markdown summary table of all experiments."""
        if not self.experiments:
            return "No experiments recorded."

        lines = [
            "| ID | Hypothesis | Decision | ROI% | Bets | Sharpe | P(ROI>0) |",
            "|---|---|---|---|---|---|---|",
        ]
        for exp in self.experiments:
            s = exp.get("summary", {})
            mc = exp.get("mc_summary", {})
            lines.append(
                f"| {exp.get('experiment_id', '?')} "
                f"| {exp.get('hypothesis', '')[:40]} "
                f"| {exp.get('decision', '?')} "
                f"| {s.get('roi_pct', 'N/A')} "
                f"| {s.get('total_bets', exp.get('n_bets', 'N/A'))} "
                f"| {s.get('sharpe_ratio', 'N/A')} "
                f"| {mc.get('prob_positive_roi', 'N/A') if mc else 'N/A'} |"
            )
        return "\n".join(lines)


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    controller = IterativeController(verbose=True)
    print("Iterative Controller initialized.")
    print(f"Experiments in registry: {len(controller.experiments)}")
    if controller.experiments:
        print("\n" + controller.summary_table())
    print("\nReady to run experiments.")
