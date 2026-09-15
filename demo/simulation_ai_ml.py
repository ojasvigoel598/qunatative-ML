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

Double verification built in:
  * MULTI-SEED  — the whole walk is repeated over several training seeds and
    results are reported as a mean, not one lucky path.
  * PER-BET LEDGER — every settled bet is written to
    backtests/results/ai_ml_bets_<tag>.csv for independent re-analysis.
  * BOOTSTRAP CI — 95% percentile interval for pooled unit ROI, resampling
    the bet ledger (2,000 resamples).  If the interval straddles 0, the ROI
    claim is not established for that agent.

Metrics: roi_pct is bankroll growth, but because the AI layer deliberately
downsizes bets, the FAIR head-to-head metric is UNIT ROI — profit per dollar
staked.  Both are reported.

Usage:
    python demo/simulation_ai_ml.py                          # I1, 5 seeds
    python demo/simulation_ai_ml.py --league E0
    python demo/simulation_ai_ml.py --train-seasons 10       # more history
    python demo/simulation_ai_ml.py --test-seasons 2425 2526 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_sim.ai_agent import AIMLBettingAgent
from agent_sim.agent import BettingAgent, RESULT_TO_OUTCOME
from agent_sim.fetch import LEAGUES
from data.real_data import SEASON_CODES, get_season

RESULTS_DIR = Path(__file__).resolve().parent.parent / "backtests" / "results"
BOOTSTRAP_RESAMPLES = 2_000


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
CLOSING_COL = {"home_win": "closing_odds_home", "draw": "closing_odds_draw",
               "away_win": "closing_odds_away"}


def run_trial(agent, world, commission: float = 0.0,
              slippage: float = 0.0) -> tuple[dict, list]:
    """One chronological pass: predict each match, then reveal the result.

    Strict point-in-time discipline:
      1. agent.decide()  — before the result is revealed
      2. agent.settle()  — after the result is revealed
      3. agent.reveal_result() — update model state only afterwards

    Realism added per bet:
      * slippage    — the price you actually get is odds * (1 - slippage)
      * commission  — exchange fee charged on net winnings (Betfair-style)
      * closing CLV — (taken odds / closing odds - 1); beating the close is
        the fastest-converging evidence that an edge is real.

    Returns (agent summary, per-bet ledger rows).
    """
    bets = []
    for _, m in world.df.iterrows():
        d = agent.decide(m, m["date"], agent.bankroll)
        agent.settle(d, m, agent.bankroll)
        if d["decision"] is not None:
            pick = d["decision"]
            won = RESULT_TO_OUTCOME[m["result"]] == pick
            odds = (d.get("odds") or {}).get(pick) or 0.0
            odds_eff = odds * (1.0 - slippage)
            profit = d["stake"] * (odds_eff - 1.0) if won else -d["stake"]
            profit_net = (profit * (1.0 - commission) if won
                          else -d["stake"])
            closing = m.get(CLOSING_COL.get(pick, ""))
            try:
                closing = float(closing)
                clv = 100.0 * (odds_eff / closing - 1.0) \
                    if closing and odds_eff else float("nan")
            except (TypeError, ValueError):
                clv = float("nan")
            bets.append({"agent": getattr(agent, "name", "ML-only"),
                         "seed": agent.seed,
                         "season": world.season,
                         "match": d["match"], "kickoff": m["date"],
                         "pick": pick, "odds": odds,
                         "odds_eff": round(odds_eff, 4),
                         "stake": d["stake"],
                         "won": int(won), "profit": round(profit, 2),
                         "profit_net": round(profit_net, 2),
                         "closing_odds": closing if closing else None,
                         "clv_pct": round(clv, 3) if clv == clv else None,
                         "ai_layer": d.get("ai_layer", "")})
        agent.reveal_result(m)
    return agent.summary(), bets


# ------------------------------------------------------------ statistics
def pooled_unit_roi(bets: list, net: bool = False) -> float:
    key = "profit_net" if net else "profit"
    staked = sum(b["stake"] for b in bets)
    return 100.0 * sum(b[key] for b in bets) / staked if staked else float("nan")


def bootstrap_unit_roi_ci(bets: list, resamples: int = BOOTSTRAP_RESAMPLES,
                          seed: int = 0, net: bool = False) -> tuple[float, float]:
    """95% percentile CI for pooled unit ROI via bet-level resampling."""
    if not bets:
        return float("nan"), float("nan")
    key = "profit_net" if net else "profit"
    profits = np.array([b[key] for b in bets], dtype=float)
    stakes = np.array([b["stake"] for b in bets], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bets), size=(resamples, len(bets)))
    rois = 100.0 * profits[idx].sum(axis=1) / stakes[idx].sum(axis=1)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def mean_clv_ci(bets: list, resamples: int = BOOTSTRAP_RESAMPLES,
                seed: int = 1) -> tuple[float, float, float]:
    """Mean per-bet closing-line CLV (%) with a 95% percentile CI.

    CLV converges far faster than ROI: ~100 bets of CLV evidence is worth
    ~10x as many bets of ROI evidence, because per-bet CLV has no result
    variance — only price variance.
    """
    clvs = np.array([b["clv_pct"] for b in bets
                     if b.get("clv_pct") is not None
                     and b["clv_pct"] == b["clv_pct"]], dtype=float)
    if len(clvs) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clvs), size=(resamples, len(clvs)))
    means = clvs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(clvs.mean()), float(lo), float(hi)


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
    parser.add_argument("--seeds", type=int, default=5,
                        help="independent training seeds (double verification)")
    parser.add_argument("--bankroll", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="first training seed; subsequent seeds increment")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tag", default="",
                        help="suffix for the results CSV (e.g. 'train5')")
    parser.add_argument("--commission", type=float, default=0.02,
                        help="exchange fee on net winnings (0.02 = Betfair "
                             "base 2%%; 0 disables)")
    parser.add_argument("--slippage", type=float, default=0.0,
                        help="fractional price deterioration vs the recorded "
                             "odds (0.01 = you get 1%% worse odds)")
    args = parser.parse_args()

    print("=" * 72)
    print("AI + ML HEAD-TO-HEAD WALK-FORWARD SIMULATION")
    print(f"League: {LEAGUES[args.league]}  Test: {','.join(args.test_seasons)}"
          f"  Train: {args.train_seasons} prior season(s)")
    print(f"Seeds: {args.seeds}  Bankroll: ${args.bankroll:,.0f}  "
          f"commission: {args.commission:.0%}  slippage: {args.slippage:.0%}")
    print("=" * 72)

    agents_cls = {"ML-only": BettingAgent, "AI+ML": AIMLBettingAgent}
    rows = {name: [] for name in agents_cls}     # per (season, seed) summaries
    bets = {name: [] for name in agents_cls}     # pooled per-bet ledger

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
        print(f"\n### Season {season}: {len(world.df)} matches, trained on "
              f"{len(prior)} season(s) ({len(train)} matches)")

        for i in range(args.seeds):
            seed = args.seed + i
            for name, cls in agents_cls.items():
                agent = cls(train, bankroll=args.bankroll, seed=seed)
                s, bet_rows = run_trial(agent, world,
                                        commission=args.commission,
                                        slippage=args.slippage)
                profit = s["final_bankroll"] - args.bankroll
                s["unit_roi_pct"] = (100.0 * profit / s["total_staked"]
                                     if s["total_staked"] else 0.0)
                s["season"], s["seed"] = season, seed
                rows[name].append(s)
                bets[name].extend(bet_rows)
            print(f"  seed {seed}: done")

    # ------------------------------------------------- aggregate + verdict
    print("\n" + "=" * 72)
    print("VERDICT — pooled over all seasons x seeds "
          "(UNIT ROI = profit / staked, the fair comparison)")
    print("=" * 72)
    verdict = {}
    for name in agents_cls:
        per_run = [r["unit_roi_pct"] for r in rows[name]]
        pooled = pooled_unit_roi(bets[name])
        pooled_net = pooled_unit_roi(bets[name], net=True)
        lo, hi = bootstrap_unit_roi_ci(bets[name])
        lo_net, hi_net = bootstrap_unit_roi_ci(bets[name], net=True)
        clv_m, clv_lo, clv_hi = mean_clv_ci(bets[name])
        n_bets = len(bets[name])
        verdict[name] = {"pooled_unit_roi_pct": pooled, "ci95": [lo, hi],
                         "pooled_net_unit_roi_pct": pooled_net,
                         "ci95_net": [lo_net, hi_net],
                         "mean_clv_pct": clv_m, "clv_ci95": [clv_lo, clv_hi],
                         "mean_run_unit_roi_pct": float(np.mean(per_run)),
                         "n_bets": n_bets}
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"  {name:<8}: pooled unit ROI {pooled:+7.2f}%  "
              f"95% CI [{lo:+.1f}%, {hi:+.1f}%] ({sig})  "
              f"mean run {np.mean(per_run):+7.2f}%  ({n_bets} bets)")
        if args.commission or args.slippage:
            sig_n = "SIGNIFICANT" if (lo_net > 0 or hi_net < 0) \
                else "not significant"
            print(f"           net of {args.commission:.0%} commission "
                  f"+ {args.slippage:.0%} slippage: {pooled_net:+7.2f}%  "
                  f"95% CI [{lo_net:+.1f}%, {hi_net:+.1f}%] ({sig_n})")
        if clv_m == clv_m:
            sig_c = "SIGNIFICANT" if (clv_lo > 0 or clv_hi < 0) \
                else "not significant"
            print(f"           closing-line CLV mean {clv_m:+.2f}%  "
                  f"95% CI [{clv_lo:+.2f}%, {clv_hi:+.2f}%] ({sig_c})  "
                  "— beats the close?")
    d = (verdict["AI+ML"]["pooled_unit_roi_pct"]
         - verdict["ML-only"]["pooled_unit_roi_pct"])
    print(f"  AI layer delta on pooled unit ROI: {d:+.2f} pts  "
          f"(vetoes {sum(r['ai_vetoes'] for r in rows['AI+ML'])}, "
          f"upscales {sum(r['ai_upscales'] for r in rows['AI+ML'])})")

    # ---------------------------------------------------------------- save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.league}" + (f"_{args.tag}" if args.tag else "")

    out = RESULTS_DIR / f"ai_ml_headtohead{suffix}.csv"
    csv_rows = []
    for name in agents_cls:
        for r in rows[name]:
            csv_rows.append({
                "agent": name, "season": r["season"], "seed": r["seed"],
                "roi_pct": r["roi_pct"], "unit_roi_pct": r["unit_roi_pct"],
                "n_bets": r["n_bets"], "n_wins": r["n_wins"],
                "strike_rate": r["strike_rate"],
                "total_staked": r["total_staked"],
                "final_bankroll": r["final_bankroll"],
                "ai_vetoes": r.get("ai_vetoes", 0),
                "ai_passes": r.get("ai_passes", 0),
                "ai_upscales": r.get("ai_upscales", 0),
                "pooled_unit_roi_pct": verdict[name]["pooled_unit_roi_pct"],
                "ci95_lo": verdict[name]["ci95"][0],
                "ci95_hi": verdict[name]["ci95"][1],
                "pooled_net_unit_roi_pct": verdict[name]["pooled_net_unit_roi_pct"],
                "ci95_net_lo": verdict[name]["ci95_net"][0],
                "ci95_net_hi": verdict[name]["ci95_net"][1],
                "mean_clv_pct": verdict[name]["mean_clv_pct"],
                "clv_ci95_lo": verdict[name]["clv_ci95"][0],
                "clv_ci95_hi": verdict[name]["clv_ci95"][1],
                "commission": args.commission, "slippage": args.slippage})
    pd.DataFrame(csv_rows).to_csv(out, index=False)

    bets_out = RESULTS_DIR / f"ai_ml_bets{suffix}.csv"
    pd.DataFrame(bets["ML-only"] + bets["AI+ML"]).to_csv(bets_out, index=False)
    print(f"\nSaved: {out}\nSaved: {bets_out}")


if __name__ == "__main__":
    main()
