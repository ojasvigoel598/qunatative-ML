#!/usr/bin/env python3
"""
FULL PROJECT EXECUTION (ML + RL)

Runs the complete quantitative sports-betting pipeline with all three layers:

    synthetic data  ->  PoissonElo + ML (Gradient Boosting) training
    ->  Q-Learning staking agent trained on a validation split
    ->  test backtest  ->  metrics + bets log + professional graphs

Usage:
    python run_full_ml_rl.py            # defaults: 1200 matches, seed 42
    python run_full_ml_rl.py --n 2000

All randomness is seeded, so the run is fully reproducible.
"""

import argparse

import pipeline


def main():
    parser = argparse.ArgumentParser(description="Quantitative Sports Betting - full ML+RL pipeline")
    parser.add_argument("--n", type=int, default=1200, help="number of synthetic matches")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerate the processed dataset")
    args = parser.parse_args()

    print("=" * 78)
    print("QUANTITATIVE SPORTS BETTING MODEL - FULL PIPELINE (PoissonElo + ML + RL)")
    print("=" * 78)

    result = pipeline.run_full_pipeline(
        n_matches=args.n, seed=args.seed,
        use_ml=True, use_rl=True, regenerate=args.regenerate, tag="ml_rl",
    )

    summary = result["summary"]
    print("\n" + "=" * 78)
    print("PROJECT COMPLETE - ALL LAYERS EXECUTED")
    print("=" * 78)
    print(pipeline.format_summary(summary))
    print("\nModel quality on the held-out test split:")
    for k, v in result["test_eval"].items():
        print(f"  {k.replace('_', ' ').title():<20}: {v}")

    # ---- two-user outputs: betting user + model user ----------------------
    from analysis.loss_attribution import write_loss_report  # noqa: E402
    from analysis.match_analysis import write_predictions_table  # noqa: E402

    test_df = result["splits"][2]
    table = write_predictions_table(
        test_df, result["models"]["poisson"], result["models"]["ml"],
        pipeline.BACKTEST_DIR / "predictions_table_ml_rl.csv",
        uncertainty_z=1.0, n_samples=150)
    report = write_loss_report(result,
                               pipeline.BACKTEST_DIR / "why_model_losing.txt")
    print("\n" + report)
    print("\nResults saved in: backtests/results/")
    print("Key files: backtest_bets_log_ml_rl.csv | metrics_ml_rl.txt | "
          "backtest_analysis_ml_rl.png | backtest_summary_ml_rl.png | "
          "predictions_table_ml_rl.csv | why_model_losing.txt")


if __name__ == "__main__":
    main()
