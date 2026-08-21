#!/usr/bin/env python3
"""
Multi-Season CLV Validation — test the CLV strategy across 4 seasons.

This script validates whether the CLV-focused strategy produces
statistically significant positive ROI across multiple seasons.

Expected outcome: If CLV truly predicts profitability, we should see
positive ROI in most seasons, with 95% CI excluding zero.

Usage:
    python scripts/21_multi_season_clv_validation.py --offline
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
# Import the engine from the previous script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "clv_engine",
    str(PROJECT_ROOT / "scripts" / "20_clv_focused_backtest.py")
)
clv_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clv_engine)
OnlineFeatureBuilder = clv_engine.OnlineFeatureBuilder
run_clv_focused_backtest = clv_engine.run_clv_focused_backtest

RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_multi_season_validation(league="SP1", verbose=True):
    """Run CLV-focused backtest across multiple seasons."""
    if verbose:
        print("=" * 70)
        print(f"MULTI-SEASON CLV VALIDATION — {LEAGUES[league]}")
        print("=" * 70)

    # Load all seasons
    frames = []
    for s in SEASON_CODES:
        try:
            frames.append(load_league(league, [s], offline=True))
        except Exception:
            continue

    if len(frames) < 3:
        print("Not enough seasons")
        return

    all_data = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"Total matches: {len(all_data)} ({all_data['date'].iloc[0].date()} -> {all_data['date'].iloc[-1].date()})")

    # Expanding window: train on seasons 1..N, test on N+1
    all_results = []

    for i in range(2, len(SEASON_CODES)):
        test_season = SEASON_CODES[i]
        train_frames = frames[:i+1]
        test_frame = frames[i]

        # Combine train data
        train_data = pd.concat(train_frames, ignore_index=True).sort_values("date").reset_index(drop=True)

        if verbose:
            print(f"\n--- Test: {SEASON_LABEL[test_season]} (train: {i+1} seasons) ---")

        # Run CLV-focused backtest
        result = run_clv_focused_backtest(
            train_data, LEAGUES[league], SEASON_LABEL[test_season],
            use_clv_filter=True,
            clv_threshold=0.0,
            odds_range=(1.6, 2.5),
            edge_threshold=0.02,
            kelly_fraction=0.20,
            verbose=False,
        )

        result["season"] = SEASON_LABEL[test_season]
        result["n_train_seasons"] = i + 1
        all_results.append(result)

        m = result["metrics"]
        if verbose:
            print(f"  Bets: {m['total_bets']:3d}  Win Rate: {m['strike_rate']:5.1f}%  "
                  f"ROI: {m['roi_pct']:+6.1f}%  CLV: {m.get('clv_mean_pct', 0):+.1f}%  "
                  f"Sharpe: {m.get('sharpe_ratio', 0):.2f}")

    # Aggregate analysis
    if all_results:
        all_metrics = []
        for r in all_results:
            m = r["metrics"].copy()
            m["season"] = r["season"]
            m["n_train_seasons"] = r["n_train_seasons"]
            all_metrics.append(m)

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(RESULTS_DIR / "multi_season_clv_results.csv", index=False)

        if verbose:
            print("\n" + "=" * 70)
            print("AGGREGATE ANALYSIS")
            print("=" * 70)

            # Overall statistics
            total_bets = metrics_df["total_bets"].sum()
            avg_roi = metrics_df["roi_pct"].mean()
            std_roi = metrics_df["roi_pct"].std()
            avg_sharpe = metrics_df["sharpe_ratio"].mean()

            print(f"  Total bets across all seasons: {total_bets}")
            print(f"  Average ROI: {avg_roi:+.1f}% (std: {std_roi:.1f}%)")
            print(f"  Average Sharpe: {avg_sharpe:.2f}")

            # Count profitable seasons
            profitable = (metrics_df["roi_pct"] > 0).sum()
            total = len(metrics_df)
            print(f"  Profitable seasons: {profitable}/{total} ({profitable/total*100:.0f}%)")

            # 95% CI for ROI
            if len(metrics_df) > 1:
                ci_low = avg_roi - 1.96 * std_roi / np.sqrt(total)
                ci_high = avg_roi + 1.96 * std_roi / np.sqrt(total)
                print(f"  95% CI for ROI: [{ci_low:+.1f}%, {ci_high:+.1f}%]")
                if ci_low > 0:
                    print("  [PASS] 95% CI excludes zero - statistically significant positive ROI")
                else:
                    print("  [FAIL] 95% CI includes zero - not statistically significant")

            # CLV analysis
            avg_clv = metrics_df["clv_mean_pct"].mean()
            print(f"  Average CLV: {avg_clv:+.1f}%")

            # Bootstrap CI for ROI
            if len(metrics_df) > 1:
                n_bootstrap = 1000
                bootstrap_rois = []
                for _ in range(n_bootstrap):
                    sample = metrics_df["roi_pct"].sample(n=len(metrics_df), replace=True)
                    bootstrap_rois.append(sample.mean())
                bootstrap_rois = np.array(bootstrap_rois)
                ci_low_boot = np.percentile(bootstrap_rois, 2.5)
                ci_high_boot = np.percentile(bootstrap_rois, 97.5)
                print(f"  Bootstrap 95% CI: [{ci_low_boot:+.1f}%, {ci_high_boot:+.1f}%]")

            # Per-season breakdown
            print("\n  Per-season breakdown:")
            for _, row in metrics_df.iterrows():
                print(f"    {row['season']}: bets={row['total_bets']:3d}  "
                      f"ROI={row['roi_pct']:+6.1f}%  "
                      f"WR={row['strike_rate']:5.1f}%  "
                      f"CLV={row.get('clv_mean_pct', 0):+.1f}%")

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Season CLV Validation")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--league", default="SP1", choices=["SP1", "E0", "I1"])
    args = parser.parse_args()

    results = run_multi_season_validation(league=args.league, verbose=True)

    print("\n[OK] Results saved to backtests/results/multi_season_clv_results.csv")


if __name__ == "__main__":
    main()
