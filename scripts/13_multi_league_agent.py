#!/usr/bin/env python3
"""
13 — Randomised multi-league walk-forward: run many independent simulations.

Each run draws a fresh scenario from its own seed (random league subset,
random walk season, random start date, random league reveal order), walks it
strictly chronologically with the ML agent, and writes its full artifacts
under backtests/results/agent_sim/.  All tables go to CSV — the terminal only
prints one-line summaries (req 10, 11).

    python scripts/13_multi_league_agent.py --seeds 100
    python scripts/13_multi_league_agent.py --seeds 5 --offline
    python scripts/13_multi_league_agent.py --seeds 3 --max-dates 80   # fast smoke

Outputs (in backtests/results/agent_sim/):
    run_<seed>_report.csv          one end-of-run report per seed (req 10)
    run_<seed>_bets.csv            complete betting ledger per seed (req 9)
    run_<seed>_opportunities.csv   every evaluated match, full audit (req 8)
    run_<seed>_nobets.csv          no-bet opportunities with reasons
    run_<seed>_by_league.csv       profit/ROI/bets per league
    run_<seed>_league_timeline.csv when each league was revealed + first bet
    run_<seed>_equity.csv          bankroll over time (sampled)
    multi_run_aggregate.csv        cross-run stats (req 11)
    multi_run_runs.csv             every per-run report row
    league_selection_frequency.csv which leagues got selected, how often
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_sim.agent import BettingAgent          # noqa: E402
from agent_sim.engine import SimulationEngine     # noqa: E402
from agent_sim.ledger import RollingLedger        # noqa: E402
from agent_sim.report import aggregate_runs, build_run_report  # noqa: E402
from agent_sim.stream import World                # noqa: E402


def run_one(seed: int, args) -> dict:
    """One full randomised walk; returns the end-of-run report dict."""
    world = World(seed=seed, leagues=args.leagues or None, offline=args.offline)
    if args.max_dates:
        world.sim_dates = world.sim_dates[: args.max_dates]
    agent = BettingAgent(world.train_df, bankroll=args.bankroll,
                         stake_mode=args.stake_mode, seed=seed)
    ledger = RollingLedger(f"run_{seed:04d}")
    engine = SimulationEngine(world, agent, ledger,
                              frozen_frac=args.frozen_frac)
    summary = engine.run()
    engine.number_bets()
    ledger.save_all()
    report_df = build_run_report(f"run_{seed:04d}", seed, world, engine, ledger)
    row = report_df.iloc[0].to_dict()
    row.update({k: summary.get(k) for k in
                ("n_bets", "n_wins", "strike_rate", "model_refits",
                 "survival")})
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=100, help="number of runs")
    p.add_argument("--start-seed", type=int, default=1000)
    p.add_argument("--bankroll", type=float, default=1_000_000.0)
    p.add_argument("--stake-mode", choices=["flat", "kelly"], default="flat")
    p.add_argument("--leagues", nargs="*", default=None,
                   help="restrict league pool (SP1 E0 D1 I1); default random")
    p.add_argument("--frozen-frac", type=float, default=0.15,
                   help="final fraction of the walk with a FROZEN model (untouched validation)")
    p.add_argument("--max-dates", type=int, default=0,
                   help="cap the walk at this many sim dates (fast smoke)")
    p.add_argument("--offline", action="store_true",
                   help="use cached data/real CSVs (no network)")
    args = p.parse_args()

    print(f"[13] multi-league walk-forward  seeds={args.seeds} "
          f"bankroll=${args.bankroll:,.0f} stake={args.stake_mode} "
          f"offline={args.offline}")
    reports, t0 = [], time.time()
    for i in range(args.seeds):
        seed = args.start_seed + i
        try:
            row = run_one(seed, args)
        except Exception as exc:                       # keep the batch alive
            print(f"  seed {seed}: FAILED ({exc})", file=sys.stderr)
            continue
        reports.append(row)
        print(f"  run {i + 1:>3}/{args.seeds} seed={seed} "
              f"leagues={row.get('leagues_analysed','')} "
              f"bets={row.get('bets_placed')} roi={row.get('roi_pct'):+.1f}% "
              f"final=${row.get('final_bankroll'):,.0f} "
              f"leaks={row.get('n_leak_flags')} refits={row.get('model_refits')}")

    if not reports:
        print("[13] no runs completed", file=sys.stderr)
        return 1
    agg = aggregate_runs(reports, [r["seed"] for r in reports])
    if len(agg):
        a = agg.iloc[0]
        print("-" * 64)
        print(f"[13] AGGREGATE over {int(a['n_runs'])} runs "
              f"({time.time() - t0:.0f}s)")
        print(f"      mean ROI {a['mean_roi']:+.2f}%   "
              f"median {a['median_roi']:+.2f}%   "
              f"std {a['std_roi']:.2f}%   "
              f"P(profit) {a['pct_profitable']:.0f}%")
        print(f"      final bankroll  mean ${a['mean_final_bankroll']:,.0f}  "
              f"median ${a['median_final_bankroll']:,.0f}")
        print(f"      worst ${a['worst_final']:,.0f}  "
              f"best ${a['best_final']:,.0f}")
        print(f"      mean max drawdown {a['mean_max_drawdown']:.1%}   "
              f"median {a['median_max_drawdown']:.1%}")
        print(f"      bets/run {a['mean_bets_per_run']:.1f}   "
              f"matches/run {a['mean_matches_evaluated']:.0f}")
        print("      -> backtests/results/agent_sim/multi_run_aggregate.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
