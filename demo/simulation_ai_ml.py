#!/usr/bin/env python3
"""
AI + ML SIMULATION — run the AI-augmented agent through real walk-forward
seasons and compare it head-to-head with the ML-only agent.

The AI layer (agent_sim/ai_agent.py) sits ON TOP of the ML model and manages
its bets using the project's proven positive-ROI signals:

  * sharp-vs-public split  (docs/13_hidden_signals.md: +1.16% CLV, t=4.72)
  * expected CLV gate      (the honest test of whether a bet has real value)
  * dispersion caution     (never size up when books disagree)
  * winner's-curse guard   (shrink stakes where ML disagrees with market most)

Both agents use the SAME underlying ML model (AdaptiveMatchPredictor), the
SAME training data (the N seasons BEFORE the test season — no leakage, and
deliberately little of it: this is the small-scale replication test) and walk
the SAME real season chronologically.  The only difference is the AI layer
filtering/sizing bets.  This isolates the value the AI reasoning adds.

Metrics: roi_pct is bankroll growth, but because the AI layer deliberately
downsizes bets, the FAIR head-to-head metric is UNIT ROI — profit per dollar
staked.  Both are reported.

Usage:
    python demo/simulation_ai_ml.py                          # I1 2526 walk
    python demo/simulation_ai_ml.py --league E0              # Premier League
    python demo/simulation_ai_ml.py --train-seasons 3        # even less data
    python demo/simulation_ai_ml.py --test-seasons 2324 2425 2526
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_sim.ai_agent import AIMLBettingAgent
from agent_sim.agent import BettingAgent
from agent_sim.fetch import LEAGUES
from data.real_data import SEASON_CODES, get_season

RESULTS_DIR = Path(__file__).resolve().parent.parent / "backtests" / "results"


# ----------------------------------------------------------------- world
class SingleLeagueWorld:
    """A simple chronological world over one league's real season.

    Exposes the same interface as agent_sim.stream.World (results_on,
    upcoming_matches, sim_dates) so the engine-like loop below works
    unchanged.  Every match uses only point-in-time information.
    """

    def __init__(self, league: str, season: str, offline: bool = False):
        self.league = league
        self.season = season
        self.df = get_season(league, season, offline)
        self.df["league_code"] = league
        self.sim_dates = sorted(pd.unique(self.df["date"]))
        self.matches_by_date = {d: [] for d in self.sim_dates}
        for _, r in self.df.iterrows():
            self.matches_by_date[r["date"]].append(r)

    def results_on(self, day) -> list:
        return list(self.matches_by_date.get(day, []))

    def upcoming_matches(self, day) -> list:
        return list(self.matches_by_date.get(day, []))


# ----------------------------------------------------------------- run
def run_trial(agent, world) -> dict:
    """One chronological pass: predict each match, then reveal the result.

    Strict point-in-time discipline:
      1. agent.decide()  — before the result is revealed
      2. agent.settle()  — after the result is revealed
      3. agent.reveal_result() — update model state only afterwards
    """
    for _, m in world.df.iterrows():
        d = agent.decide(m, m["date"], agent.bankroll)
        agent.settle(d, m, agent.bankroll)
        agent.reveal_result(m)
    return agent.summary()


def make_agents(train_df: pd.DataFrame, bankroll: float, seed: int) -> dict:
    """Fresh ML-only and AI+ML agents trained on identical (small) history."""
    return {
        "ML-only": BettingAgent(train_df, bankroll=bankroll, seed=seed),
        "AI+ML": AIMLBettingAgent(train_df, bankroll=bankroll, seed=seed),
    }


def main():
    parser = argparse.ArgumentParser(
        description="AI+ML vs ML-only head-to-head walk-forward simulation")
    parser.add_argument("--league", default="I1",
                        choices=list(LEAGUES.keys()))
    parser.add_argument("--test-seasons", nargs="+", default=["2425", "2526"],
                        choices=SEASON_CODES,
                        help="season(s) to walk forward over (default: the "
                             "two most recent — one season is too small a sample)")
    parser.add_argument("--train-seasons", type=int, default=5,
                        help="how many prior seasons of ONE league the agents "
                             "may learn from (small-scale replication test)")
    parser.add_argument("--bankroll", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tag", default="",
                        help="suffix for the results CSV (e.g. 'train5')")
    args = parser.parse_args()

    print("=" * 72)
    print("AI + ML HEAD-TO-HEAD WALK-FORWARD SIMULATION")
    print(f"League: {LEAGUES[args.league]}  Test: {','.join(args.test_seasons)}"
          f"  Train: {args.train_seasons} prior season(s)")
    print(f"Bankroll: ${args.bankroll:,.0f}  Seed: {args.seed}")
    print("=" * 72)

    # rows accumulate across test seasons; each season is an independent trial
    rows = {name: [] for name in ("ML-only", "AI+ML")}
    for season in args.test_seasons:
        # ---- training history = the N seasons immediately BEFORE the test
        ti = SEASON_CODES.index(season)
        prior = SEASON_CODES[max(0, ti - args.train_seasons):ti]
        if not prior:
            sys.exit(f"[FAIL] no prior seasons before {season}")
        train = pd.concat(
            [get_season(args.league, s, args.offline) for s in prior],
            ignore_index=True)
        train["league_code"] = args.league

        world = SingleLeagueWorld(args.league, season, args.offline)
        print(f"\n### Season {season}: {len(world.df)} matches, "
              f"trained on {len(prior)} season(s) ({len(train)} matches)\n")

        for name, agent in make_agents(train, args.bankroll, args.seed).items():
            print(f"--- {name} agent ---")
            s = run_trial(agent, world)
            profit = s["final_bankroll"] - args.bankroll
            unit_roi = (100.0 * profit / s["total_staked"]
                        if s["total_staked"] else 0.0)
            s["unit_roi_pct"] = unit_roi
            s["season"] = season
            rows[name].append(s)
            for k in ("roi_pct", "unit_roi_pct", "n_bets", "n_wins",
                      "total_staked", "final_bankroll"):
                v = s[k]
                print(f"  {k:<14}: {v:,.2f}" if isinstance(v, float)
                      else f"  {k:<14}: {v}")
            if name == "AI+ML":
                print(f"  {'ai_vetoes':<14}: {s['ai_vetoes']}  "
                      f"passes: {s['ai_passes']}  upscales: {s['ai_upscales']}")

    # ------------------------------------------------- aggregate + verdict
    def agg(name, key):
        vals = [r[key] for r in rows[name]]
        return float(np.mean(vals)) if vals else float("nan")

    def agg_unit_roi(name):
        staked = sum(r["total_staked"] for r in rows[name])
        profit = sum(r["final_bankroll"] - args.bankroll for r in rows[name])
        return 100.0 * profit / staked if staked else float("nan")

    print("\n" + "=" * 72)
    print(f"VERDICT (mean over {len(args.test_seasons)} season(s); "
          f"UNIT ROI = profit / staked — the fair comparison)")
    print("=" * 72)
    for name in ("ML-only", "AI+ML"):
        n_bets = sum(r["n_bets"] for r in rows[name])
        staked = sum(r["total_staked"] for r in rows[name])
        print(f"  {name:<8}: bankroll ROI {agg(name, 'roi_pct'):+.2f}%  "
              f"UNIT ROI {agg_unit_roi(name):+.2f}%  "
              f"({n_bets} bets, ${staked:,.0f} staked)")
    ml_u, ai_u = agg_unit_roi("ML-only"), agg_unit_roi("AI+ML")
    print(f"  AI layer delta on unit ROI: {ai_u - ml_u:+.2f} pts  "
          f"(vetoes {sum(r['ai_vetoes'] for r in rows['AI+ML'])}, "
          f"upscales {sum(r['ai_upscales'] for r in rows['AI+ML'])})")

    # ---------------------------------------------------------------- save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / ("ai_ml_headtohead"
                         + (f"_{args.tag}" if args.tag else "") + ".csv")
    csv_rows = []
    for name in ("ML-only", "AI+ML"):
        for r in rows[name]:
            csv_rows.append({
                "agent": name, "season": r["season"],
                "roi_pct": r["roi_pct"], "unit_roi_pct": r["unit_roi_pct"],
                "n_bets": r["n_bets"], "n_wins": r["n_wins"],
                "strike_rate": r["strike_rate"],
                "total_staked": r["total_staked"],
                "final_bankroll": r["final_bankroll"],
                "ai_vetoes": r.get("ai_vetoes", 0),
                "ai_passes": r.get("ai_passes", 0),
                "ai_upscales": r.get("ai_upscales", 0)})
    pd.DataFrame(csv_rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
