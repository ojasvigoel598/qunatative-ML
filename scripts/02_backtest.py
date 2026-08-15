#!/usr/bin/env python3
"""
Backtest script - runs the train/validation/test backtest on the processed
dataset and saves metrics + plots to backtests/results/.

This script delegates to the shared engine in pipeline.py so the CLI, the
demos and the notebook all use identical logic.

Usage:
    python scripts/02_backtest.py                # PoissonElo + Kelly (base)
    python scripts/02_backtest.py --ml-rl         # PoissonElo + ML + RL
    python scripts/02_backtest.py --n 2000        # custom sample size
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the backtest")
    parser.add_argument("--ml-rl", action="store_true",
                        help="use the full PoissonElo + ML + RL pipeline")
    parser.add_argument("--n", type=int, default=1200, help="synthetic sample size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerate the processed dataset")
    args = parser.parse_args()

    df, _ = pipeline.load_or_generate_data(n_matches=args.n, seed=args.seed,
                                           regenerate=args.regenerate)

    tag = "ml_rl" if args.ml_rl else ""
    result = pipeline.run_backtest(
        df, use_ml=args.ml_rl, use_rl=args.ml_rl, seed=args.seed,
        tag=tag, save_results=True, verbose=True)

    print("\n" + "=" * 60)
    print(pipeline.format_summary(result["summary"]))
    print("\nBacktest complete. Check backtests/results/")


if __name__ == "__main__":
    main()
