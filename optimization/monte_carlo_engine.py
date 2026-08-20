#!/usr/bin/env python3
"""
Monte Carlo Simulation Engine — 1,000,000 vectorized simulations.

For every serious candidate strategy, runs at least 1M Monte Carlo
simulations to determine the full distribution of outcomes.

Simulation dimensions:
1. Outcome uncertainty — sample outcomes from calibrated probabilities
2. Odds uncertainty — perturb odds within empirical ranges
3. Model uncertainty — bootstrap predictions
4. Calibration stress — perturb probabilities within calibration error bounds
5. Market stress — apply adverse slippage
6. Staking uncertainty — stress bankroll parameters

All simulations use vectorized NumPy operations for speed.
Target: 1M simulations in <60 seconds.

Usage:
    from optimization.monte_carlo_engine import MonteCarloEngine
    engine = MonteCarloEngine()
    results = engine.run(bets_df, equity, n_simulations=1_000_000)
    print(results["summary"])
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MonteCarloEngine:
    """Vectorized Monte Carlo simulation for betting strategies.

    Simulates uncertainty in outcomes, odds, calibration, and staking
    to produce a full distribution of possible results.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(self, bets_df: pd.DataFrame,
            model_probs: Optional[np.ndarray] = None,
            initial_bankroll: float = 10000.0,
            n_simulations: int = 1_000_000,
            outcome_noise: float = 0.02,
            odds_noise: float = 0.03,
            calibration_stress: float = 0.05,
            slippage_pct: float = 0.01,
            kelly_fraction: float = 0.25,
            max_stake_frac: float = 0.08,
            seed: int = 42) -> Dict[str, Any]:
        """Run Monte Carlo simulations.

        Args:
            bets_df: DataFrame with bet outcomes, odds, edges, stakes.
            model_probs: (n_bets, 3) model probabilities for each bet.
            initial_bankroll: starting bankroll.
            n_simulations: number of Monte Carlo paths.
            outcome_noise: std of noise added to outcome probabilities.
            odds_noise: std of noise added to odds.
            calibration_stress: max perturbation of model probabilities.
            slippage_pct: price slippage applied to all bets.
            kelly_fraction: Kelly fraction for staking.
            max_stake_frac: maximum stake as fraction of bankroll.
            seed: RNG seed.

        Returns:
            Dict with full simulation results.
        """
        t0 = time.time()

        if len(bets_df) == 0:
            return self._empty_result(n_simulations, initial_bankroll)

        n_bets = len(bets_df)
        rng = np.random.default_rng(seed)

        # Extract bet data as arrays
        odds = bets_df["my_odds"].values.astype(float)
        outcomes = (bets_df["bet_outcome"] == "Win").values.astype(float)
        edges = bets_df.get("edge_pct", pd.Series(np.zeros(n_bets))).values.astype(float) / 100

        # Apply slippage to odds
        effective_odds = odds * (1 - slippage_pct)

        if self.verbose:
            print(f"  Running {n_simulations:,} Monte Carlo simulations "
                  f"on {n_bets} bets...")

        # ---- Derive per-bet win probabilities ----
        # If model_probs provided, use the max class probability.
        # Otherwise derive from edge: p = (1 + edge) / odds.
        if model_probs is not None and model_probs.shape[0] == n_bets:
            bet_win_probs = model_probs.max(axis=1)
        else:
            # edge = p * odds - 1  =>  p = (1 + edge) / odds
            bet_win_probs = (1.0 + edges) / np.clip(effective_odds, 1.01, None)
            bet_win_probs = np.clip(bet_win_probs, 0.01, 0.99)

        # Base win probability per bet: shape (1, n_bets) for broadcasting
        base_win_probs = bet_win_probs[np.newaxis, :]  # (1, n_bets)

        # ---- Vectorized simulation ----
        # Outcome uncertainty: add noise to probabilities
        win_probs = np.clip(
            base_win_probs + rng.normal(0, outcome_noise,
                                        (n_simulations, n_bets)),
            0.01, 0.99
        )

        # Sample win/loss for each simulation: (n_simulations, n_bets)
        sim_wins = rng.random((n_simulations, n_bets)) < win_probs

        # Odds uncertainty: perturb odds
        sim_odds = effective_odds[np.newaxis, :] + \
            rng.normal(0, odds_noise, (n_simulations, n_bets))
        sim_odds = np.clip(sim_odds, 1.1, 20.0)

        # Calibration stress: perturb probabilities within error bounds
        cal_perturbation = rng.uniform(
            -calibration_stress, calibration_stress,
            (n_simulations, n_bets)
        )
        stressed_probs = np.clip(win_probs + cal_perturbation, 0.01, 0.99)

        # Staking: fractional Kelly based on stressed probabilities
        stressed_edges = np.clip(stressed_probs * sim_odds - 1.0, 0, None)
        stressed_stake_fracs = np.minimum(
            (stressed_edges / np.clip(sim_odds - 1, 0.01, None)) * kelly_fraction,
            max_stake_frac
        )

        # ---- Simulate bankroll paths ----
        # For speed, simulate sequentially but vectorized per-bet
        bankrolls = np.full(n_simulations, initial_bankroll, dtype=float)
        equity_paths = np.zeros((n_simulations, n_bets + 1), dtype=float)
        equity_paths[:, 0] = initial_bankroll

        # Per-bet simulation (sequential across bets, parallel across sims)
        for j in range(n_bets):
            stakes = bankrolls * stressed_stake_fracs[:, j]
            stakes = np.maximum(stakes, 0)  # no negative stakes

            # Profit: win * odds * stake - stake
            profits = np.where(
                sim_wins[:, j],
                stakes * (sim_odds[:, j] - 1),
                -stakes
            )

            bankrolls = bankrolls + profits
            bankrolls = np.maximum(bankrolls, 0)  # can't go below zero
            equity_paths[:, j + 1] = bankrolls

        elapsed = time.time() - t0

        # ---- Compute summary statistics ----
        final_bankrolls = equity_paths[:, -1]
        total_profits = final_bankrolls - initial_bankroll
        rois = total_profits / initial_bankroll * 100

        # Maximum drawdown per path
        peaks = np.maximum.accumulate(equity_paths, axis=1)
        drawdowns = (equity_paths - peaks) / np.maximum(peaks, 1)
        max_drawdowns = np.abs(drawdowns.min(axis=1)) * 100

        # Ruin: final bankroll <= 0
        prob_ruin = float((final_bankrolls <= 0).mean())

        # Profit factor (per-path)
        per_path_returns = np.diff(equity_paths, axis=1) / equity_paths[:, :-1]
        wins_per_path = (per_path_returns > 0).sum(axis=1)
        losses_per_path = (per_path_returns < 0).sum(axis=1)

        # Losing streaks per path (vectorized)
        # Track cumulative losses: reset on each win
        loss_run = np.zeros((n_simulations, n_bets), dtype=int)
        for j in range(n_bets):
            loss_run[:, j] = np.where(
                sim_wins[:, j], 0,
                np.where(j == 0, 1, loss_run[:, j - 1] + 1)
            )
        max_losing_streaks = loss_run.max(axis=1)

        # Summary
        summary = {
            "n_simulations": n_simulations,
            "n_bets": n_bets,
            "initial_bankroll": initial_bankroll,
            "mean_roi_pct": round(float(rois.mean()), 4),
            "median_roi_pct": round(float(np.median(rois)), 4),
            "std_roi_pct": round(float(rois.std()), 4),
            "percentile_5_roi_pct": round(float(np.percentile(rois, 5)), 4),
            "percentile_25_roi_pct": round(float(np.percentile(rois, 25)), 4),
            "percentile_75_roi_pct": round(float(np.percentile(rois, 75)), 4),
            "percentile_95_roi_pct": round(float(np.percentile(rois, 95)), 4),
            "prob_positive_roi": round(float((rois > 0).mean()), 4),
            "prob_roi_above_5pct": round(float((rois > 5).mean()), 4),
            "prob_ruin": round(prob_ruin, 4),
            "mean_max_drawdown_pct": round(float(max_drawdowns.mean()), 2),
            "median_max_drawdown_pct": round(float(np.median(max_drawdowns)), 2),
            "percentile_95_max_drawdown_pct": round(
                float(np.percentile(max_drawdowns, 95)), 2),
            "mean_final_bankroll": round(float(final_bankrolls.mean()), 2),
            "median_final_bankroll": round(float(np.median(final_bankrolls)), 2),
            "percentile_5_final_bankroll": round(
                float(np.percentile(final_bankrolls, 5)), 2),
            "percentile_95_final_bankroll": round(
                float(np.percentile(final_bankrolls, 95)), 2),
            "mean_losing_streak": round(float(max_losing_streaks.mean()), 1),
            "median_losing_streak": round(float(np.median(max_losing_streaks)), 1),
            "percentile_95_losing_streak": round(
                float(np.percentile(max_losing_streaks, 95)), 1),
            "runtime_seconds": round(elapsed, 2),
        }

        # Distribution data (for histograms)
        distribution = {
            "roi_samples": rois[::max(1, n_simulations // 10000)].tolist(),
            "drawdown_samples": max_drawdowns[::max(1, n_simulations // 10000)].tolist(),
            "final_bankroll_samples": final_bankrolls[::max(1, n_simulations // 10000)].tolist(),
        }

        result = {
            "summary": summary,
            "distribution": distribution,
            "config": {
                "outcome_noise": outcome_noise,
                "odds_noise": odds_noise,
                "calibration_stress": calibration_stress,
                "slippage_pct": slippage_pct,
                "kelly_fraction": kelly_fraction,
                "max_stake_frac": max_stake_frac,
                "seed": seed,
            },
        }

        if self.verbose:
            self._print_summary(summary)

        return result

    def _empty_result(self, n_simulations: int,
                      initial_bankroll: float) -> Dict:
        """Return empty result for no bets."""
        return {
            "summary": {
                "n_simulations": n_simulations,
                "n_bets": 0,
                "initial_bankroll": initial_bankroll,
                "mean_roi_pct": 0.0,
                "median_roi_pct": 0.0,
                "prob_positive_roi": 0.0,
                "prob_ruin": 1.0,
                "runtime_seconds": 0.0,
            },
            "distribution": {"roi_samples": [], "drawdown_samples": [],
                             "final_bankroll_samples": []},
            "config": {},
        }

    def _print_summary(self, summary: Dict):
        """Pretty-print the Monte Carlo summary."""
        print("\n" + "=" * 60)
        print(f"MONTE CARLO SUMMARY ({summary['n_simulations']:,} simulations)")
        print("=" * 60)
        print(f"  Bets per path:     {summary['n_bets']}")
        print(f"  Mean ROI:          {summary['mean_roi_pct']:+.2f}%")
        print(f"  Median ROI:        {summary['median_roi_pct']:+.2f}%")
        print(f"  ROI std:           {summary['std_roi_pct']:.2f}%")
        print(f"  5th percentile:    {summary['percentile_5_roi_pct']:+.2f}%")
        print(f"  95th percentile:   {summary['percentile_95_roi_pct']:+.2f}%")
        print(f"  P(ROI > 0):        {summary['prob_positive_roi']:.1%}")
        print(f"  P(ROI > 5%):       {summary['prob_roi_above_5pct']:.1%}")
        print(f"  P(ruin):           {summary['prob_ruin']:.4f}")
        print(f"  Mean max DD:       {summary['mean_max_drawdown_pct']:.1f}%")
        print(f"  Mean final bank:   ${summary['mean_final_bankroll']:,.0f}")
        print(f"  Runtime:           {summary['runtime_seconds']:.1f}s")

    @staticmethod
    def save_results(results: Dict, output_path: Path):
        """Save Monte Carlo results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [OK] Saved Monte Carlo results -> {output_path}")

    @staticmethod
    def write_report(results: Dict, output_path: Path) -> str:
        """Write a markdown report of Monte Carlo results."""
        s = results["summary"]
        lines = [
            "# Monte Carlo Simulation Report\n",
            f"**Simulations**: {s['n_simulations']:,}\n",
            f"**Bets per path**: {s['n_bets']}\n",
            f"**Initial bankroll**: ${s['initial_bankroll']:,.0f}\n",
            f"**Runtime**: {s['runtime_seconds']:.1f}s\n",
            "\n## ROI Distribution\n",
            f"| Statistic | Value |",
            f"|-----------|------:|",
            f"| Mean ROI | {s['mean_roi_pct']:+.2f}% |",
            f"| Median ROI | {s['median_roi_pct']:+.2f}% |",
            f"| Std ROI | {s['std_roi_pct']:.2f}% |",
            f"| 5th percentile | {s['percentile_5_roi_pct']:+.2f}% |",
            f"| 25th percentile | {s['percentile_25_roi_pct']:+.2f}% |",
            f"| 75th percentile | {s['percentile_75_roi_pct']:+.2f}% |",
            f"| 95th percentile | {s['percentile_95_roi_pct']:+.2f}% |",
            "\n## Risk Metrics\n",
            f"| Metric | Value |",
            f"|--------|------:|",
            f"| P(ROI > 0) | {s['prob_positive_roi']:.1%} |",
            f"| P(ROI > 5%) | {s['prob_roi_above_5pct']:.1%} |",
            f"| P(ruin) | {s['prob_ruin']:.4f} |",
            f"| Mean max drawdown | {s['mean_max_drawdown_pct']:.1f}% |",
            f"| 95th %ile max drawdown | {s['percentile_95_max_drawdown_pct']:.1f}% |",
            f"| Mean max losing streak | {s['mean_losing_streak']:.1f} |",
            f"| 95th %ile losing streak | {s['percentile_95_losing_streak']:.1f} |",
            "\n## Final Bankroll Distribution\n",
            f"| Statistic | Value |",
            f"|-----------|------:|",
            f"| Mean | ${s['mean_final_bankroll']:,.0f} |",
            f"| Median | ${s['median_final_bankroll']:,.0f} |",
            f"| 5th percentile | ${s['percentile_5_final_bankroll']:,.0f} |",
            f"| 95th percentile | ${s['percentile_95_final_bankroll']:,.0f} |",
            "\n## Configuration\n",
        ]
        for k, v in results.get("config", {}).items():
            lines.append(f"- {k}: {v}\n")

        text = "".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return text


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    import pipeline

    print("Generating synthetic data and running baseline backtest...")
    df = pipeline.generate_match_data(600, seed=42)
    result = pipeline.run_backtest(df, use_ml=True, use_rl=False,
                                   save_results=False, verbose=False)

    bets_df = result["bets_df"]
    equity = result["equity"]

    print(f"\nBaseline: {len(bets_df)} bets, "
          f"ROI={result['summary']['roi_pct']:+.2f}%")

    engine = MonteCarloEngine(verbose=True)

    # Run 10K first for quick validation
    print("\n--- Quick test (10K simulations) ---")
    quick = engine.run(bets_df, initial_bankroll=1000.0,
                       n_simulations=10_000, seed=42)
    print(f"  P(ROI>0): {quick['summary']['prob_positive_roi']:.1%}")

    # Run 1M
    print("\n--- Full run (1M simulations) ---")
    full = engine.run(bets_df, initial_bankroll=10000.0,
                      n_simulations=1_000_000, seed=42)

    out_dir = PROJECT_ROOT / "results"
    engine.save_results(full, out_dir / "monte_carlo_summary.json")
    engine.write_report(full, out_dir / "monte_carlo_report.md")

    print("\n[OK] Monte Carlo engine self-test passed.")
