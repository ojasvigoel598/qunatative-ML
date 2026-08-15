#!/usr/bin/env python3
"""
FULL PROJECT EXECUTION (BASE)

Runs the complete quantitative sports-betting pipeline with the core model:

    synthetic data  ->  PoissonElo training  ->  backtest (fractional Kelly)
    ->  metrics + bets log + professional graphs

Usage:
    python run_full_project.py            # defaults: 1200 matches, seed 42
    python run_full_project.py --n 2000   # custom sample size

Everything is seeded, so the run is fully reproducible.
"""

import argparse

import pipeline


def main():
    parser = argparse.ArgumentParser(description="Quantitative Sports Betting - base pipeline")
    parser.add_argument("--n", type=int, default=1200, help="number of synthetic matches")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerate the processed dataset")
    args = parser.parse_args()

    print("=" * 70)
    print("QUANTITATIVE SPORTS BETTING MODEL - FULL PROJECT (PoissonElo + Kelly)")
    print("=" * 70)

    result = pipeline.run_full_pipeline(
        n_matches=args.n, seed=args.seed,
        use_ml=False, use_rl=False, regenerate=args.regenerate, tag="",
    )

    summary = result["summary"]
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE")
    print("=" * 70)
    print(pipeline.format_summary(summary))
    print("\nResults saved in: backtests/results/")
    print("Key files: backtest_bets_log.csv | metrics.txt | "
          "backtest_analysis.png | backtest_summary.png")


if __name__ == "__main__":
    main()
