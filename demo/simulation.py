#!/usr/bin/env python3
"""
$1,000,000 INVESTMENT SIMULATION with the trained model.

This script answers the question: *"If I invested $1,000,000 following the
trained model, how much would I end up with?"*

How it works
------------
1. The full pipeline (PoissonElo + Gradient Boosting + Q-Learning staking) is
   trained once on the historical dataset (the same models used by
   run_full_ml_rl.py).
2. A forward Monte Carlo simulation is then run: for each trial, a fresh stream
   of future matches is generated in the *same* synthetic world (same team
   strengths, fresh fixtures and bookmaker lines).
3. For every match the trained ensemble predicts probabilities, looks for
   value (edge > 3% on a confident outcome), sizes the stake with the trained
   Q-Learning agent (falling back to quarter-Kelly), and the result is sampled
   from the TRUE match probabilities.
4. Bankroll starts at $1,000,000 each trial.  The distribution of final
   bankrolls across trials is reported, together with a chart.

Why Monte Carlo?
----------------
A single backtest path is dominated by variance (see README).  Running many
independent forward paths turns the single gamble into a distribution, which
is the honest way to summarise expected performance.

IMPORTANT
---------
This is a synthetic, calibrated demonstration.  The model is "betting" against
a simulated bookmaker inside its own world.  Results reflect the world's
assumptions and are NOT a prediction of real-world returns.

Run:
    python demo/simulation.py
    python demo/simulation.py --trials 50 --matches 1500
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
from models.rl_staking_agent import QLearningStakingAgent  # noqa: E402

SIM_OUT = Path(__file__).resolve().parent / "output"
SIM_OUT.mkdir(parents=True, exist_ok=True)

INITIAL_INVESTMENT = 1_000_000.0
EDGE_THRESHOLD = pipeline.EDGE_THRESHOLD
MIN_ODDS = pipeline.MIN_ODDS
MIN_MODEL_PROB = pipeline.MIN_MODEL_PROB
MIN_STAKE = pipeline.MIN_STAKE


def _world_strengths(seed: int = 42) -> dict:
    """Recover the latent team strengths of the training world (deterministic)."""
    rng = np.random.default_rng(seed)
    return {t: float(rng.normal(0, 1)) for t in pipeline.TEAMS}


def _forward_match_stream(strengths: dict, rng: np.random.Generator, n: int):
    """Yield (home, away, p_true, opening_odds, closing_odds) for n future matches."""
    for _ in range(n):
        home = str(rng.choice(pipeline.TEAMS))
        away = str(rng.choice([t for t in pipeline.TEAMS if t != home]))
        diff = strengths[home] - strengths[away]
        lam_home = 1.6 * np.exp(0.22 * diff) * 1.12
        lam_away = 1.3 * np.exp(-0.22 * diff)
        p_true = pipeline._true_probs(float(lam_home), float(lam_away))
        margin = float(rng.uniform(0.05, 0.08))
        opening, closing = pipeline._make_bookie_odds(
            p_true, rng, margin,
            prob_noise=pipeline.BOOKIE_PROB_NOISE, gamma=pipeline.BOOKIE_GAMMA)
        yield home, away, p_true, opening, closing


def _prediction_table(poisson, ml) -> dict:
    """Pre-compute model probabilities for every ordered team pair.

    The model is frozen after training (Elo does not update in-sample), so a
    fixture's prediction depends only on the (home, away) pair - 10 teams give
    90 ordered pairs instead of 1200 per-trial predictions.
    """
    table = {}
    for home in pipeline.TEAMS:
        for away in pipeline.TEAMS:
            if home == away:
                continue
            table[(home, away)] = pipeline.ensemble_probs(poisson, ml, home, away)
    return table


FLAT_STAKE = 10_000.0   # flat staking: $10K per bet (variance-minimising policy)


def run_trial(poisson, ml, rl_agent, strengths, rng, n_matches: int,
              policy: str = "flat") -> tuple:
    """Simulate one forward path.  Returns (final_bankroll, equity_path, n_bets).

    policy="flat"  (default, from the 100-trial stress test): $10K per bet,
                   no compounding - lowest ruin probability.
    policy="kelly" : quarter-Kelly / RL-scaled stakes (original behaviour).
    """
    bankroll = INITIAL_INVESTMENT
    equity = [bankroll]
    n_bets = 0
    prob_table = _prediction_table(poisson, ml)

    for home, away, p_true, opening, closing in _forward_match_stream(strengths, rng, n_matches):
        probs = prob_table[(home, away)]
        edges = poisson.calculate_edge(probs, opening, threshold=EDGE_THRESHOLD)
        best = edges.get("best_value")
        if not best or edges.get("max_edge", 0) < EDGE_THRESHOLD:
            continue
        odds = opening[best]
        if odds < MIN_ODDS or probs[best] < MIN_MODEL_PROB:
            continue
        edge = edges[best]

        if policy == "flat":
            stake = min(FLAT_STAKE, bankroll)
        else:
            if rl_agent is not None:
                stake_frac = rl_agent.get_stake_fraction(edge, odds, bankroll, INITIAL_INVESTMENT)
                if stake_frac <= 0:
                    stake_frac = pipeline._fractional_kelly(edge, odds)
            else:
                stake_frac = pipeline._fractional_kelly(edge, odds)
            stake = bankroll * stake_frac
        if stake < MIN_STAKE:
            continue

        # Outcome sampled from the TRUE distribution (the world, not the model).
        p_win = p_true[best]
        win = rng.random() < p_win
        bankroll += stake * (odds - 1.0) if win else -stake
        n_bets += 1
        equity.append(bankroll)

    return bankroll, np.array(equity), n_bets


def main():
    parser = argparse.ArgumentParser(description="$1M investment simulation with the trained model")
    parser.add_argument("--trials", type=int, default=25,
                        help="number of Monte Carlo trials (default 25)")
    parser.add_argument("--matches", type=int, default=1200,
                        help="forward matches per trial (default 1200, ~3.3 years)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="save chart + CSV to demo/output/")
    parser.add_argument("--policy", choices=["flat", "kelly"], default="flat",
                        help="staking policy (default flat=$10K per bet - the "
                             "variance-minimising policy from the stress test)")
    args = parser.parse_args()

    print("=" * 78)
    print("SIMULATION: IF YOU INVESTED $1,000,000, WHAT WOULD THE TRAINED MODEL DO?")
    print(f"  Staking policy: {args.policy} "
          + ("(flat $10K per bet)" if args.policy == "flat" else "(quarter-Kelly)"))
    print("=" * 78)

    # 1. Train the exact same models used by run_full_ml_rl.py
    print("\n[1/3] Training the model layers on historical data (seed 42)...")
    df, _ = pipeline.load_or_generate_data(n_matches=1200, seed=42)
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: int(n * 0.65)].copy()
    valid_df = df.iloc[int(n * 0.65): int(n * 0.80)].copy()
    poisson, ml = pipeline.train_models(train_df, use_ml=True, verbose=True)

    # Train the RL staking agent on the validation discovery backtest
    experiences = pipeline._discovery_experiences(valid_df, poisson, ml, 1000.0)
    rl_agent = QLearningStakingAgent()
    rl_agent.train(experiences, episodes=200)

    # 2. Monte Carlo forward simulation
    print(f"\n[2/3] Running {args.trials} forward trials of {args.matches} matches each...")
    strengths = _world_strengths(42)
    final_bankrolls = []
    bet_counts = []
    sample_paths = []

    for t in range(args.trials):
        rng = np.random.default_rng(1000 + t)
        final, equity, n_bets = run_trial(poisson, ml, rl_agent, strengths, rng,
                                          args.matches, policy=args.policy)
        final_bankrolls.append(final)
        bet_counts.append(n_bets)
        if t < 8:
            sample_paths.append(equity)

    finals = np.array(final_bankrolls)
    profits = finals - INITIAL_INVESTMENT
    years = args.matches / 365.25
    median_cagr = (np.median(finals) / INITIAL_INVESTMENT) ** (1 / years) - 1

    # 3. Report
    print("\n[3/3] Results")
    print("-" * 60)
    print(f"  Initial investment      : ${INITIAL_INVESTMENT:,.0f}")
    print(f"  Forward horizon         : {args.matches:,} matches (~{years:.1f} years)")
    print(f"  Trials                  : {args.trials}")
    print(f"  Bets per trial (avg)    : {np.mean(bet_counts):.0f}")
    print()
    print(f"  Final bankroll - mean   : ${finals.mean():,.0f}")
    print(f"  Final bankroll - median : ${np.median(finals):,.0f}")
    print(f"  Expected profit         : ${profits.mean():,.0f}  ({profits.mean() / INITIAL_INVESTMENT * 100:+.1f}%)")
    print(f"  P(end with profit)      : {(profits > 0).mean() * 100:.0f}%")
    print(f"  90% range               : [${np.percentile(finals, 5):,.0f} , ${np.percentile(finals, 95):,.0f}]")
    print(f"  Best / worst trial      : ${finals.max():,.0f} / ${finals.min():,.0f}")
    print(f"  Median CAGR             : {median_cagr * 100:+.1f}% per year")
    print("-" * 60)

    if args.save:
        pd.DataFrame({
            "trial": np.arange(args.trials) + 1,
            "final_bankroll": finals,
            "profit": profits,
            "n_bets": bet_counts,
        }).to_csv(SIM_OUT / "simulation_1m_trials.csv", index=False)

        _plot_simulation(finals, profits, sample_paths, years)
        print(f"  [OK] Saved {SIM_OUT / 'simulation_1m.png'}")
        print(f"  [OK] Saved {SIM_OUT / 'simulation_1m_trials.csv'}")

    print("\n" + "=" * 78)
    print("BOTTOM LINE")
    print("=" * 78)
    print(f"  Investing $1,000,000 with the trained model over ~{years:.0f} years of")
    print(f"  simulated matches ends, on average, at ${finals.mean():,.0f} "
          f"({profits.mean() / INITIAL_INVESTMENT * 100:+.1f}%),")
    print(f"  with a {np.median(finals):,.0f} median outcome and "
          f"{(profits > 0).mean() * 100:.0f}% probability of finishing in profit.")
    print()
    print("  This is a Monte Carlo simulation inside a synthetic world - it shows")
    print("  how the pipeline behaves, not a prediction of real-world returns.")
    print("=" * 78)


def _plot_simulation(finals: np.ndarray, profits: np.ndarray,
                     sample_paths: list, years: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].hist(finals / 1e6, bins=20, color="#2E86AB", edgecolor="white")
    axes[0].axvline(1.0, color="gray", linestyle="--", linewidth=2,
                    label="Initial $1M")
    axes[0].axvline(np.median(finals) / 1e6, color="#EF476F", linestyle="-",
                    linewidth=2, label=f"Median ${np.median(finals) / 1e6:,.2f}M")
    axes[0].set_title(f"Distribution of Final Bankroll ($1M invested, {years:.0f} years)")
    axes[0].set_xlabel("Final bankroll ($M)")
    axes[0].set_ylabel("Trials")
    axes[0].legend()

    for eq in sample_paths:
        axes[1].plot(eq / 1e6, alpha=0.6, linewidth=1.0)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=2)
    axes[1].set_title("Sample Equity Paths (first 8 trials)")
    axes[1].set_xlabel("Bets placed")
    axes[1].set_ylabel("Bankroll ($M)")

    plt.tight_layout()
    plt.savefig(SIM_OUT / "simulation_1m.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
