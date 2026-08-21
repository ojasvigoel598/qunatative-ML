#!/usr/bin/env python3
"""
Tight CLV Filter — test different CLV thresholds to find optimal edge.

Hypothesis: Higher CLV thresholds should produce higher ROI per bet
but fewer total bets. The optimal threshold maximizes ROI while
maintaining sufficient sample size.

Usage:
    python scripts/22_tight_clv_filter.py --offline
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.real_data import load_league, LEAGUES, SEASON_CODES, SEASON_LABEL

RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import the CLV backtest engine
import importlib.util
spec = importlib.util.spec_from_file_location(
    "clv_engine",
    str(PROJECT_ROOT / "scripts" / "20_clv_focused_backtest.py")
)
clv_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clv_engine)
OnlineFeatureBuilder = clv_engine.OnlineFeatureBuilder
run_clv_focused_backtest = clv_engine.run_clv_focused_backtest


def test_clv_thresholds(league="SP1", verbose=True):
    """Test different CLV thresholds to find optimal edge."""
    if verbose:
        print("=" * 70)
        print(f"CLV THRESHOLD OPTIMIZATION — {LEAGUES[league]}")
        print("=" * 70)

    # Load all data
    frames = []
    for s in SEASON_CODES:
        try:
            frames.append(load_league(league, [s], offline=True))
        except Exception:
            continue

    all_data = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    # Test different CLV thresholds
    thresholds = [0.0, 2.0, 4.0]
    results = []

    for threshold in thresholds:
        if verbose:
            print(f"\n--- CLV threshold: {threshold:+.1f}% ---")

        # Run backtest with this threshold
        result = run_clv_focused_backtest(
            all_data, LEAGUES[league], "All Seasons",
            use_clv_filter=True,
            clv_threshold=threshold / 100,  # convert to decimal
            odds_range=(1.6, 2.5),
            edge_threshold=0.02,
            kelly_fraction=0.20,
            verbose=False,
        )

        m = result["metrics"]
        results.append({
            "threshold_pct": threshold,
            "total_bets": m["total_bets"],
            "roi_pct": m["roi_pct"],
            "strike_rate": m["strike_rate"],
            "avg_edge_pct": m["avg_edge_pct"],
            "clv_mean_pct": m.get("clv_mean_pct", 0),
            "sharpe_ratio": m.get("sharpe_ratio", 0),
            "profit_factor": m.get("profit_factor", 0),
        })

        if verbose:
            print(f"  Bets: {m['total_bets']:3d}  ROI: {m['roi_pct']:+6.1f}%  "
                  f"WR: {m['strike_rate']:5.1f}%  Sharpe: {m.get('sharpe_ratio', 0):.2f}")

    # Find optimal threshold
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "clv_threshold_results.csv", index=False)

    if verbose:
        print("\n" + "=" * 70)
        print("THRESHOLD COMPARISON")
        print("=" * 70)
        print(results_df.to_string(index=False))

        # Optimal by ROI
        best_roi = results_df.loc[results_df["roi_pct"].idxmax()]
        print(f"\n  Best by ROI: CLV >={best_roi['threshold_pct']:.1f}% "
              f"(ROI: {best_roi['roi_pct']:+.1f}%, Bets: {best_roi['total_bets']})")

        # Optimal by Sharpe
        best_sharpe = results_df.loc[results_df["sharpe_ratio"].idxmax()]
        print(f"  Best by Sharpe: CLV >={best_sharpe['threshold_pct']:.1f}% "
              f"(Sharpe: {best_sharpe['sharpe_ratio']:.2f}, ROI: {best_sharpe['roi_pct']:+.1f}%)")

        # Optimal by profit factor
        best_pf = results_df.loc[results_df["profit_factor"].idxmax()]
        print(f"  Best by Profit Factor: CLV >={best_pf['threshold_pct']:.1f}% "
              f"(PF: {best_pf['profit_factor']:.2f}, ROI: {best_pf['roi_pct']:+.1f}%)")

    return results_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLV Threshold Optimization")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--league", default="SP1", choices=["SP1", "E0", "I1"])
    args = parser.parse_args()

    results = test_clv_thresholds(league=args.league, verbose=True)

    print("\n[OK] Results saved to backtests/results/clv_threshold_results.csv")


if __name__ == "__main__":
    main()
