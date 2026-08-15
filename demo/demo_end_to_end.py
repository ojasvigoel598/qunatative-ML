#!/usr/bin/env python3
"""
END-TO-END DEMO of the Quantitative Sports Betting Model.

This script walks the entire project exactly as a user would run it, printing
a narrated, step-by-step trace and saving every output (bets log, metrics,
plots) to demo/output/.  It is the companion script to demo/simulation.py
(which answers "if I invested $1M, what would the trained model do?").

Run:
    python demo/demo_end_to_end.py
    python demo/demo_end_to_end.py --quick      # smaller demo world
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402

DEMO_OUT = Path(__file__).resolve().parent / "output"
DEMO_OUT.mkdir(parents=True, exist_ok=True)

SEPARATOR = "=" * 78


def banner(title: str):
    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)


def step(n: int, title: str):
    print(f"\n[STEP {n}] {title}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="End-to-end demo of the sports betting model")
    parser.add_argument("--quick", action="store_true",
                        help="use a smaller world (600 matches) for a fast demo")
    parser.add_argument("--n", type=int, default=None,
                        help="number of synthetic matches (overrides --quick)")
    args = parser.parse_args()

    n_matches = args.n or (600 if args.quick else 1200)

    print(SEPARATOR)
    print("QUANTITATIVE SPORTS BETTING MODEL - END-TO-END DEMO")
    print("Poisson + Elo  |  Gradient Boosting ML layer  |  Q-Learning staking")
    print(SEPARATOR)

    # ---------------------------------------------------------------- 1. Data
    step(1, "Data: build a realistic synthetic football world")
    print("  * 10 teams, each with a latent strength")
    print("  * goals drawn from a Poisson process (home advantage included)")
    print("  * bookmaker odds derived from the TRUE probabilities plus")
    print("    a margin and the well-documented favourite-longshot bias")
    print("  * closing odds are drawn independently (used for CLV)")
    df, generated = pipeline.load_or_generate_data(
        n_matches=n_matches, seed=42, regenerate=args.quick or args.n is not None)
    print(f"\n  Dataset: {len(df)} matches")
    print(f"  Columns: {', '.join(df.columns)}")
    print(df.head(3).to_string(index=False))

    # ---------------------------------------------------------------- 2. Split
    step(2, "Train / validation / test split (65 / 15 / 20)")
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: int(n * 0.65)].copy()
    valid_df = df.iloc[int(n * 0.65): int(n * 0.80)].copy()
    test_df = df.iloc[int(n * 0.80):].copy()
    print(f"  Train: {len(train_df)} | Validation: {len(valid_df)} | Test: {len(test_df)}")
    print("  The test split is never touched during training or validation.")

    # ---------------------------------------------------------------- 3. Models
    step(3, "Train the model layers (on the TRAIN split only)")
    poisson, ml = pipeline.train_models(train_df, use_ml=True, verbose=True)
    print("\n  Sample prediction (test fixture, model sees only Elo + form):")
    r = test_df.iloc[0]
    probs = pipeline.ensemble_probs(poisson, ml, r["home_team"], r["away_team"])
    fair = poisson.probs_to_fair_odds(probs)
    print(f"  {r['home_team']} vs {r['away_team']}: P(home)={probs['home_win']:.2f} "
          f"P(draw)={probs['draw']:.2f} P(away)={probs['away_win']:.2f}")

    # -------------------------------------------------- 4. Value / RL training
    step(4, "Discover value bets on VALIDATION and train the RL staking agent")
    experiences = pipeline._discovery_experiences(valid_df, poisson, ml, 1000.0)
    print(f"  Kelly discovery backtest on validation: {len(experiences)} realized bets")
    from models.rl_staking_agent import QLearningStakingAgent
    rl_agent = QLearningStakingAgent()
    rl_agent.train(experiences, episodes=200)
    print("  The agent learned stake multipliers relative to the quarter-Kelly baseline.")

    # ---------------------------------------------------------------- 5. Backtest
    step(5, "Run the TEST backtest (PoissonElo + ML ensemble + RL staking)")
    res = pipeline.run_backtest(
        df, use_ml=True, use_rl=True, seed=42, tag="demo", save_results=True,
        out_dir=DEMO_OUT, verbose=True)

    summary = res["summary"]
    bets_df = res["bets_df"]
    equity = res["equity"]

    # ---------------------------------------------------------------- 6. Results
    step(6, "Results")
    print(pipeline.format_summary(summary))
    print("\n  Model quality on the held-out test split (all 240 matches, not just bets):")
    for k, v in res["test_eval"].items():
        print(f"    {k.replace('_', ' ').title():<20}: {v}")

    print("\n  First 5 bets in the log:")
    cols = ["date", "match", "market", "my_odds", "stake", "edge_pct",
            "bet_outcome", "profit_loss", "clv_pct", "running_bankroll"]
    print(bets_df[cols].head(5).to_string(index=False))

    # ---------------------------------------------------------------- 7. Save
    step(7, "Save everything for the demo record")
    log = pd.DataFrame({
        "stage": ["data", "train", "validation", "test", "final"],
        "n_matches": [len(df), len(train_df), len(valid_df), len(test_df), len(bets_df)],
        "detail": [
            f"{n_matches} synthetic matches, seed 42",
            f"PoissonElo + ML trained",
            f"{len(experiences)} RL experiences",
            f"{summary['total_bets']} value bets",
            f"final bankroll ${summary['final_bankroll']:,.2f} (ROI {summary['roi_pct']}%)",
        ],
    })
    log.to_csv(DEMO_OUT / "demo_log.csv", index=False)
    equity_series = pd.Series(equity, name="bankroll")
    equity_series.to_csv(DEMO_OUT / "demo_equity.csv", index=False)
    print(f"  [OK] Saved {DEMO_OUT / 'demo_log.csv'}")
    print(f"  [OK] Saved {DEMO_OUT / 'demo_equity.csv'}")
    print(f"  [OK] Saved {DEMO_OUT / 'backtest_bets_log_demo.csv'}")
    print(f"  [OK] Saved {DEMO_OUT / 'metrics_demo.txt'}")
    print(f"  [OK] Saved {DEMO_OUT / 'backtest_analysis_demo.png'}")
    print(f"  [OK] Saved {DEMO_OUT / 'backtest_summary_demo.png'}")

    # ---------------------------------------------------------------- 8. Wrap-up
    banner("DEMO COMPLETE")
    print(f"  Final bankroll : ${summary['final_bankroll']:,.2f}")
    print(f"  ROI            : {summary['roi_pct']}%")
    print(f"  Strike rate    : {summary['strike_rate']}%")
    print(f"  Sharpe         : {summary['sharpe_ratio']}")
    print(f"  Max drawdown   : {summary['max_drawdown_pct']}%")
    print(f"  Model accuracy : {res['test_eval']['accuracy']:.1%} "
          f"(baseline {res['test_eval']['baseline_accuracy']:.1%})")
    print("\n  Next: python demo/simulation.py  ->  what happens if you invest $1M")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
