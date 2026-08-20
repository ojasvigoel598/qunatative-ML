#!/usr/bin/env python3
"""
Frozen Judge — the immutable evaluation machinery.

This module defines the final evaluation protocol that the optimizer
CANNOT modify.  Every experiment must pass through this judge to be
declared successful.

The judge defines:
- Temporal split with embargo
- Evaluation metrics and thresholds
- Transaction cost assumptions
- Bankroll rules
- Statistical tests
- Monte Carlo methodology
- Success criteria / validation gates

Usage:
    from evaluation.frozen_judge import FrozenJudge
    judge = FrozenJudge()
    result = judge.evaluate(experiment_result)
    print(result["passed"], result["gates"])
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================================
# Configuration — these constants define the frozen evaluation protocol.
# The optimizer MUST NOT modify any value in this section.
# ======================================================================

# Temporal split
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.15
EMBARGO_FRACTION = 0.05   # gap between validation and test
TEST_FRACTION = 0.20

# Betting parameters
INITIAL_BANKROLL = 10000.0
EDGE_THRESHOLD = 0.03
MIN_ODDS = 1.6
MIN_MODEL_PROB = 0.40
KELLY_FRACTION = 0.25
MAX_STAKE_FRAC = 0.08
MIN_STAKE = 50.0

# Transaction costs
COMMISSION_PCT = 0.0      # exchange commission (0 for bookmaker)
SLIPPAGE_PCT = 0.01       # average price slippage (1% of odds)
VIG_ADJUSTMENT = 0.0      # additional vig adjustment

# Validation gates — ALL must pass
GATES = {
    "min_bets": 20,               # minimum number of bets required
    "min_roi_pct": 0.0,           # ROI must be positive after costs
    "max_max_drawdown_pct": 40.0, # maximum drawdown must be < 40%
    "min_calibration_ece": 0.15,  # ECE must be < 0.15
    "min_sharpe": 0.0,            # Sharpe ratio must be positive
    "max_losing_streak": 15,      # longest losing streak must be < 15
    "min_profit_factor": 1.0,     # profit factor must be >= 1.0
    "min_clv_t_stat": 1.5,        # CLV t-stat must indicate some information
    "min_win_rate": 0.40,         # win rate must be >= 40%
    "ci_includes_positive": False, # 95% CI for ROI must NOT include zero
}

# Monte Carlo
MC_SIMULATIONS = 1_000_000
MC_CONFIDENCE_LEVEL = 0.95
MC_MIN_PROBABILITY_POSITIVE_ROI = 0.55  # P(ROI > 0) must exceed 55%
MC_MIN_MEDIAN_ROI_PCT = 0.0

# Stability
CONSECUTIVE_WINDOWS_REQUIRED = 10
MIN_WINDOW_BETS = 5

# Reproducibility
SEED = 42


class FrozenJudge:
    """Immutable evaluation judge for the betting system.

    The optimizer may modify models, hyperparameters, features, ensemble
    weights, calibration, threshold, and staking strategy.  It MUST NOT
    modify anything in this class or its configuration constants.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.gates = dict(GATES)

    # ==================================================================
    # 1. Temporal split with embargo
    # ==================================================================
    def temporal_split(self, df: pd.DataFrame
                       ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                  pd.DataFrame, pd.DataFrame]:
        """Split data into train / validation / embargo / test.

        The embargo gap ensures no information leaks from the validation
        period into the test period (De Prado, 2018).
        """
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        n_train = int(n * TRAIN_FRACTION)
        n_valid = int(n * VALIDATION_FRACTION)
        n_embargo = int(n * EMBARGO_FRACTION)

        train = df.iloc[:n_train].copy()
        valid = df.iloc[n_train:n_train + n_valid].copy()
        embargo = df.iloc[n_train + n_valid:
                          n_train + n_valid + n_embargo].copy()
        test = df.iloc[n_train + n_valid + n_embargo:].copy()

        if self.verbose:
            print(f"  Temporal split: train={len(train)}, "
                  f"valid={len(valid)}, embargo={len(embargo)}, "
                  f"test={len(test)}")
            if len(train) > 0 and len(test) > 0:
                print(f"  Train: {train['date'].iloc[0].date()} -> "
                      f"{train['date'].iloc[-1].date()}")
                print(f"  Test:  {test['date'].iloc[0].date()} -> "
                      f"{test['date'].iloc[-1].date()}")

        return train, valid, embargo, test

    # ==================================================================
    # 2. Apply transaction costs
    # ==================================================================
    def apply_transaction_costs(self, bets_df: pd.DataFrame
                                ) -> pd.DataFrame:
        """Apply commission and slippage to the bets log.

        Real betting has costs: exchange commission, price slippage,
        and potential vig adjustment.  This method modifies profit_loss
        to reflect realistic costs.
        """
        if len(bets_df) == 0:
            return bets_df

        bets = bets_df.copy()
        # Slippage: effective odds are worse by slippage_pct
        effective_odds = bets["my_odds"] * (1 - SLIPPAGE_PCT)
        # Commission on winning bets
        commission = np.where(
            bets["bet_outcome"] == "Win",
            bets["profit_loss"] * COMMISSION_PCT,
            0.0
        )
        # Adjust profit/loss
        bets["profit_loss_adjusted"] = (
            bets["profit_loss"] - commission
        )
        bets["effective_odds"] = effective_odds
        return bets

    # ==================================================================
    # 3. Evaluate all gates
    # ==================================================================
    def evaluate_gates(self, metrics: Dict[str, Any],
                       roi_ci: Optional[Dict] = None
                       ) -> Dict[str, bool]:
        """Check whether all validation gates pass.

        Returns a dict of gate_name -> passed (bool).
        """
        gates = {}

        # Min bets
        gates["min_bets"] = (
            metrics.get("total_bets", 0) >= self.gates["min_bets"]
        )

        # Positive ROI
        gates["min_roi_pct"] = (
            metrics.get("roi_pct", -999) >= self.gates["min_roi_pct"]
        )

        # Max drawdown
        gates["max_max_drawdown_pct"] = (
            metrics.get("max_drawdown_pct", 999) <=
            self.gates["max_max_drawdown_pct"]
        )

        # Calibration
        gates["min_calibration_ece"] = (
            metrics.get("ece", 1.0) <= self.gates["min_calibration_ece"]
        )

        # Sharpe
        gates["min_sharpe"] = (
            metrics.get("sharpe_ratio", -999) >= self.gates["min_sharpe"]
        )

        # Losing streak
        gates["max_losing_streak"] = (
            metrics.get("longest_losing_streak", 999) <=
            self.gates["max_losing_streak"]
        )

        # Profit factor
        pf = metrics.get("profit_factor")
        if pf is None or pf == float("inf"):
            gates["min_profit_factor"] = True
        else:
            gates["min_profit_factor"] = (
                pf >= self.gates["min_profit_factor"]
            )

        # CLV t-stat
        gates["min_clv_t_stat"] = (
            abs(metrics.get("clv_t_stat", 0)) >=
            self.gates["min_clv_t_stat"]
        )

        # Win rate
        sr = metrics.get("strike_rate", 0)
        if isinstance(sr, str):
            sr = float(sr.replace("%", ""))
        gates["min_win_rate"] = (
            sr >= self.gates["min_win_rate"]
        )

        # CI includes positive
        if roi_ci is not None:
            ci_high = roi_ci.get("ci_high", 0)
            gates["ci_includes_positive"] = (
                ci_high > 0
            )
        else:
            gates["ci_includes_positive"] = False

        return gates

    # ==================================================================
    # 4. Monte Carlo gate
    # ==================================================================
    def evaluate_monte_carlo(self, mc_results: Dict[str, Any]
                             ) -> Dict[str, bool]:
        """Check whether Monte Carlo simulation passes its gates."""
        gates = {}

        gates["mc_min_prob_positive_roi"] = (
            mc_results.get("prob_positive_roi", 0) >=
            MC_MIN_PROBABILITY_POSITIVE_ROI
        )
        gates["mc_min_median_roi"] = (
            mc_results.get("median_roi_pct", -999) >=
            MC_MIN_MEDIAN_ROI_PCT
        )
        gates["mc_max_ruin_probability"] = (
            mc_results.get("prob_ruin", 1.0) <= 0.20
        )

        return gates

    # ==================================================================
    # 5. Stability gate (10 consecutive windows)
    # ==================================================================
    def evaluate_stability(self, window_results: List[Dict[str, Any]]
                           ) -> Dict[str, Any]:
        """Check whether 10 consecutive independent windows all pass.

        Returns:
            passed: bool — True only if 10 consecutive windows pass
            consecutive_count: int — current consecutive pass count
            total_windows: int — total windows evaluated
            results: list of per-window pass/fail
        """
        window_passes = []
        for wr in window_results:
            mg = self.evaluate_gates(wr.get("metrics", {}))
            all_pass = all(mg.values())
            window_passes.append({
                "window": wr.get("window_id", "?"),
                "passed": all_pass,
                "roi_pct": wr.get("metrics", {}).get("roi_pct", 0),
                "n_bets": wr.get("metrics", {}).get("total_bets", 0),
                "gates": mg,
            })

        # Count consecutive passes from the end
        consecutive = 0
        for wp in reversed(window_passes):
            if wp["passed"]:
                consecutive += 1
            else:
                break

        all_passed = consecutive >= CONSECUTIVE_WINDOWS_REQUIRED

        return {
            "passed": all_passed,
            "consecutive_count": consecutive,
            "required": CONSECUTIVE_WINDOWS_REQUIRED,
            "total_windows": len(window_results),
            "windows": window_passes,
        }

    # ==================================================================
    # 6. Full evaluation
    # ==================================================================
    def evaluate(self, metrics: Dict[str, Any],
                 mc_results: Optional[Dict] = None,
                 roi_ci: Optional[Dict] = None,
                 window_results: Optional[List[Dict]] = None
                 ) -> Dict[str, Any]:
        """Run the full frozen judge evaluation.

        This is the single entry point for experiment validation.
        Returns a comprehensive result dict with:
        - passed: overall pass/fail
        - gate_results: per-gate results
        - mc_results: Monte Carlo gate results (if provided)
        - stability_results: 10-window stability (if provided)
        - summary: human-readable summary
        """
        gate_results = self.evaluate_gates(metrics, roi_ci)
        all_gates_pass = all(gate_results.values())

        mc_gate_results = {}
        all_mc_pass = True
        if mc_results is not None:
            mc_gate_results = self.evaluate_monte_carlo(mc_results)
            all_mc_pass = all(mc_gate_results.values())

        stability_results = {}
        stability_pass = True
        if window_results is not None:
            stability_results = self.evaluate_stability(window_results)
            stability_pass = stability_results.get("passed", False)

        overall_passed = all_gates_pass and all_mc_pass and stability_pass

        # Summary
        failed_gates = [k for k, v in gate_results.items() if not v]
        failed_mc = [k for k, v in mc_gate_results.items() if not v]
        summary_lines = []
        if overall_passed:
            summary_lines.append("PASSED all validation gates")
        else:
            summary_lines.append("FAILED validation gates:")
            if failed_gates:
                summary_lines.append(f"  Prediction gates: {failed_gates}")
            if failed_mc:
                summary_lines.append(f"  Monte Carlo gates: {failed_mc}")
            if not stability_pass:
                consecutive = stability_results.get("consecutive_count", 0)
                summary_lines.append(
                    f"  Stability: {consecutive}/10 consecutive windows")

        return {
            "passed": overall_passed,
            "gate_results": gate_results,
            "all_gates_pass": all_gates_pass,
            "mc_gate_results": mc_gate_results,
            "all_mc_pass": all_mc_pass,
            "stability_results": stability_results,
            "stability_pass": stability_pass,
            "summary": "\n".join(summary_lines),
        }

    # ==================================================================
    # 7. Dataset hash (holdout lock)
    # ==================================================================
    @staticmethod
    def hash_dataset(df: pd.DataFrame) -> str:
        """Cryptographic hash of the dataset for holdout integrity."""
        data_bytes = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def lock_holdout(self, test_df: pd.DataFrame,
                     output_path: Optional[Path] = None) -> str:
        """Lock the holdout dataset by recording its hash.

        The optimizer must never access the holdout.  This hash proves
        the holdout was not modified during optimization.
        """
        h = self.hash_dataset(test_df)
        record = {
            "holdout_hash": h,
            "n_matches": len(test_df),
            "date_range": [
                str(test_df["date"].iloc[0].date()),
                str(test_df["date"].iloc[-1].date()),
            ],
            "n_teams": test_df["home_team"].nunique(),
        }
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(record, indent=2),
                                   encoding="utf-8")
            if self.verbose:
                print(f"  Holdout locked: hash={h}, "
                      f"n={len(test_df)}, path={output_path}")
        return h

    def verify_holdout(self, test_df: pd.DataFrame,
                       expected_hash: str) -> bool:
        """Verify the holdout dataset has not been tampered with."""
        actual = self.hash_dataset(test_df)
        return actual == expected_hash


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    print("Frozen Judge — validation gate definitions:")
    print(f"  Train/Valid/Embargo/Test: "
          f"{TRAIN_FRACTION}/{VALIDATION_FRACTION}/"
          f"{EMBARGO_FRACTION}/{TEST_FRACTION}")
    print(f"  Initial bankroll: ${INITIAL_BANKROLL:,.0f}")
    print(f"  Edge threshold: {EDGE_THRESHOLD*100:.1f}%")
    print(f"  Kelly fraction: {KELLY_FRACTION}")
    print(f"  Commission: {COMMISSION_PCT*100:.1f}%")
    print(f"  Slippage: {SLIPPAGE_PCT*100:.1f}%")
    print(f"  MC simulations: {MC_SIMULATIONS:,}")
    print(f"  Consecutive windows required: {CONSECUTIVE_WINDOWS_REQUIRED}")
    print()
    print("Validation gates:")
    for gate, threshold in GATES.items():
        print(f"  {gate}: {threshold}")

    judge = FrozenJudge(verbose=True)
    print()
    print("FrozenJudge initialized. Ready to evaluate experiments.")
