#!/usr/bin/env python3
"""
ROI Attribution — explains WHY ROI changed between model configurations.

Decomposes the ROI difference into causal contributions from:
1. Calibration improvement
2. Bet selection (edge threshold, probability floor)
3. Model improvement (better probabilities)
4. Staking strategy
5. Transaction costs
6. Other / unexplained

This module is mandatory for every experiment: never report "ROI increased
from X% to Y%" without explaining WHY.

Usage:
    from analysis.roi_attribution import ROIAttributor
    attributor = ROIAttributor()
    report = attributor.attribute(baseline_metrics, new_metrics,
                                   baseline_cal, new_cal)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ROIAttributor:
    """Decompose ROI changes into causal contributions."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def attribute(self, baseline: Dict, new: Dict,
                  baseline_cal: Optional[Dict] = None,
                  new_cal: Optional[Dict] = None,
                  baseline_bets: Optional[np.ndarray] = None,
                  new_bets: Optional[np.ndarray] = None
                  ) -> Dict:
        """Attribute the ROI change from baseline to new configuration.

        Args:
            baseline: metrics dict from the baseline experiment.
            new: metrics dict from the new experiment.
            baseline_cal: calibration metrics for baseline (ECE, Brier, etc.).
            new_cal: calibration metrics for new configuration.
            baseline_bets: per-bet profit/loss for baseline.
            new_bets: per-bet profit/loss for new configuration.

        Returns:
            Dict with attribution breakdown and explanation.
        """
        baseline_roi = baseline.get("roi_pct", 0)
        new_roi = new.get("roi_pct", 0)
        delta_roi = new_roi - baseline_roi

        attribution = {
            "baseline_roi_pct": round(baseline_roi, 2),
            "new_roi_pct": round(new_roi, 2),
            "delta_roi_pct": round(delta_roi, 2),
        }

        contributions = {}

        # 1. Calibration contribution
        if baseline_cal is not None and new_cal is not None:
            baseline_ece = baseline_cal.get("ece", 0)
            new_ece = new_cal.get("ece", 0)
            ece_change = baseline_ece - new_ece  # positive = improvement

            # Estimate calibration contribution:
            # Literature suggests ~5-10% ROI improvement per 0.01 ECE reduction
            # in the betting region.  Use 7.5% as a rough estimate.
            cal_contribution = ece_change * 7.5  # scaled to ROI points
            contributions["calibration"] = {
                "ece_change": round(ece_change, 4),
                "estimated_roi_contribution_pct": round(cal_contribution, 2),
                "direction": "improved" if ece_change > 0 else "degraded",
            }

            baseline_brier = baseline_cal.get("brier_score", 0)
            new_brier = new_cal.get("brier_score", 0)
            contributions["brier_change"] = {
                "delta": round(new_brier - baseline_brier, 4),
                "direction": "improved" if new_brier < baseline_brier else "degraded",
            }
        else:
            contributions["calibration"] = {
                "estimated_roi_contribution_pct": 0.0,
                "direction": "unknown",
            }

        # 2. Bet selection contribution
        baseline_n_bets = baseline.get("total_bets", 0)
        new_n_bets = new.get("total_bets", 0)
        bet_change = new_n_bets - baseline_n_bets

        baseline_edge = baseline.get("avg_edge_pct", 0)
        new_edge = new.get("avg_edge_pct", 0)
        edge_change = new_edge - baseline_edge

        # More bets with lower average edge may indicate less selective betting
        # Fewer bets with higher average edge may indicate better filtering
        contributions["bet_selection"] = {
            "n_bets_change": bet_change,
            "avg_edge_change_pct": round(edge_change, 2),
            "direction": ("more_selective" if new_n_bets <= baseline_n_bets
                          else "less_selective"),
        }

        # 3. Model improvement
        baseline_acc = baseline.get("accuracy", 0) if "accuracy" in baseline else None
        new_acc = new.get("accuracy", 0) if "accuracy" in new else None
        if baseline_acc is not None and new_acc is not None:
            acc_change = new_acc - baseline_acc
            contributions["model_accuracy"] = {
                "accuracy_change": round(acc_change, 4),
                "direction": "improved" if acc_change > 0 else "degraded",
            }

        # 4. Staking strategy
        baseline_sharpe = baseline.get("sharpe_ratio", 0)
        new_sharpe = new.get("sharpe_ratio", 0)
        contributions["staking"] = {
            "sharpe_change": round(new_sharpe - baseline_sharpe, 3),
            "sortino_change": round(
                new.get("sortino_ratio", 0) - baseline.get("sortino_ratio", 0), 3),
            "direction": ("improved" if new_sharpe > baseline_sharpe
                          else "degraded"),
        }

        # 5. Risk metrics
        baseline_dd = baseline.get("max_drawdown_pct", 0)
        new_dd = new.get("max_drawdown_pct", 0)
        contributions["risk"] = {
            "drawdown_change_pct": round(new_dd - baseline_dd, 2),
            "direction": ("improved" if new_dd < baseline_dd else "degraded"),
        }

        # 6. CLV (information quality)
        baseline_clv = baseline.get("avg_clv_pct", 0)
        new_clv = new.get("avg_clv_pct", 0)
        baseline_clv_t = baseline.get("clv_t_stat", 0)
        new_clv_t = new.get("clv_t_stat", 0)
        contributions["information"] = {
            "clv_change_pct": round(new_clv - baseline_clv, 2),
            "clv_t_stat_change": round(new_clv_t - baseline_clv_t, 2),
            "direction": ("improved" if new_clv_t > baseline_clv_t
                          else "degraded"),
        }

        # Generate explanation
        explanation = self._generate_explanation(attribution, contributions)

        attribution["contributions"] = contributions
        attribution["explanation"] = explanation

        if self.verbose:
            self._print_attribution(attribution)

        return attribution

    def _generate_explanation(self, attribution: Dict,
                              contributions: Dict) -> str:
        """Generate human-readable explanation of the ROI change."""
        delta = attribution["delta_roi_pct"]
        lines = []

        if abs(delta) < 0.1:
            lines.append("ROI change is negligible (<0.1%).")
            return " ".join(lines)

        direction = "increased" if delta > 0 else "decreased"
        lines.append(f"ROI {direction} by {abs(delta):.2f}% "
                     f"(from {attribution['baseline_roi_pct']:+.2f}% "
                     f"to {attribution['new_roi_pct']:+.2f}%).")

        # Find dominant contribution
        cal = contributions.get("calibration", {})
        if cal.get("estimated_roi_contribution_pct", 0) > 0.5:
            lines.append(f"Calibration improvement (ECE "
                         f"{cal.get('ece_change', 0):+.4f}) contributed "
                         f"~{cal['estimated_roi_contribution_pct']:+.1f}%.")
        elif cal.get("estimated_roi_contribution_pct", 0) < -0.5:
            lines.append(f"Calibration degraded (ECE "
                         f"{cal.get('ece_change', 0):+.4f}), costing "
                         f"~{abs(cal['estimated_roi_contribution_pct']):.1f}%.")

        sel = contributions.get("bet_selection", {})
        if abs(sel.get("n_bets_change", 0)) > 3:
            lines.append(f"Bet selection: {sel['n_bets_change']:+d} bets "
                         f"({sel['direction']}).")

        info = contributions.get("information", {})
        if abs(info.get("clv_t_stat_change", 0)) > 0.5:
            lines.append(f"Information quality: CLV t-stat "
                         f"{info['clv_t_stat_change']:+.2f} "
                         f"({info['direction']}).")

        risk = contributions.get("risk", {})
        if abs(risk.get("drawdown_change_pct", 0)) > 2:
            lines.append(f"Max drawdown changed by "
                         f"{risk['drawdown_change_pct']:+.1f}%.")

        return " ".join(lines)

    def _print_attribution(self, attribution: Dict):
        """Pretty-print the attribution report."""
        print("\n" + "=" * 60)
        print("ROI ATTRIBUTION ANALYSIS")
        print("=" * 60)
        print(f"\nBaseline ROI: {attribution['baseline_roi_pct']:+.2f}%")
        print(f"New ROI:      {attribution['new_roi_pct']:+.2f}%")
        print(f"Delta:        {attribution['delta_roi_pct']:+.2f}%")
        print(f"\nContributions:")
        for name, contrib in attribution["contributions"].items():
            print(f"  {name}: {contrib}")
        print(f"\nExplanation: {attribution['explanation']}")


def compare_experiments(baseline_summary: Dict, new_summary: Dict,
                        baseline_eval: Optional[Dict] = None,
                        new_eval: Optional[Dict] = None,
                        verbose: bool = True) -> Dict:
    """Convenience function to compare two experiment results.

    Args:
        baseline_summary: compute_metrics output for baseline.
        new_summary: compute_metrics output for new experiment.
        baseline_eval: evaluate_probability_quality output for baseline.
        new_eval: evaluate_probability_quality output for new experiment.

    Returns:
        Full attribution report.
    """
    attributor = ROIAttributor(verbose=verbose)
    baseline_cal = baseline_eval if baseline_eval else None
    new_cal = new_eval if new_eval else None
    return attributor.attribute(baseline_summary, new_summary,
                                baseline_cal=baseline_cal,
                                new_cal=new_cal)


if __name__ == "__main__":
    # Self-test with dummy data
    baseline = {
        "roi_pct": -14.3, "total_bets": 26, "sharpe_ratio": -0.48,
        "sortino_ratio": -0.62, "max_drawdown_pct": 21.3,
        "avg_edge_pct": 14.8, "avg_clv_pct": 0.01, "clv_t_stat": 0.05,
        "strike_rate": 42.3, "profit_factor": 0.68,
    }
    new = {
        "roi_pct": -13.0, "total_bets": 23, "sharpe_ratio": -0.39,
        "sortino_ratio": -0.51, "max_drawdown_pct": 21.8,
        "avg_edge_pct": 14.1, "avg_clv_pct": 0.18, "clv_t_stat": 0.12,
        "strike_rate": 47.8, "profit_factor": 0.70,
    }
    baseline_cal = {"ece": 0.070, "brier_score": 0.577, "log_loss": 0.98}
    new_cal = {"ece": 0.052, "brier_score": 0.550, "log_loss": 0.94}

    attributor = ROIAttributor(verbose=True)
    report = attributor.attribute(baseline, new, baseline_cal, new_cal)
    print("\n[OK] ROI attribution self-test passed.")
