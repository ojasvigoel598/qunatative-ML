#!/usr/bin/env python3
"""
SEASON-TO-DATE SIMULATION — bet every played match of the current real season
(default: Premier League 2026/27) with both agents, chronologically, showing
the AI layer's verdict on EVERY match.

For each played match the table shows:
  * the actual score,
  * the ML model's pick and probability (trained ONLY on prior seasons),
  * the stake the agent put on it,
  * the AI verdict: PASS (with stake) or VETO (with the reason),
  * the settled result and money won/lost.

Training uses only the N prior seasons of the same league (default 20).
No match from the test season is ever seen before its kickoff.

Usage:
    python demo/season_to_date_sim.py --offline
    python demo/season_to_date_sim.py --offline --league E0 --season 2627
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_sim.ai_agent import AIMLBettingAgent
from agent_sim.agent import BettingAgent, RESULT_TO_OUTCOME
from agent_sim.fetch import LEAGUES
from data.real_data import SEASON_CODES, get_season

RESULTS_DIR = Path(__file__).resolve().parent.parent / "backtests" / "results"


def bucket(reason: str, pick) -> str:
    """Collapse the many micro-reasons into a small histogram."""
    if pick:
        return "BET PLACED"
    if reason.startswith("AI VETO"):
        return "AI veto"
    if "no edge" in reason or "no positive edge" in reason:
        return "ML: no edge vs market"
    if "floor" in reason:
        return "ML: prob below floor"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="E0", choices=list(LEAGUES.keys()))
    ap.add_argument("--season", default="2627", choices=SEASON_CODES)
    ap.add_argument("--train-seasons", type=int, default=20)
    ap.add_argument("--bankroll", type=float, default=1_000_000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    ti = SEASON_CODES.index(args.season)
    prior = SEASON_CODES[max(0, ti - args.train_seasons):ti]
    train = pd.concat([get_season(args.league, s, args.offline)
                       for s in prior], ignore_index=True)
    train["league_code"] = args.league
    test = get_season(args.league, args.season, args.offline)
    played = test.dropna(subset=["result"]).reset_index(drop=True)
    played["league_code"] = args.league
    print("=" * 96)
    print(f"SEASON-TO-DATE SIMULATION — {LEAGUES[args.league]} "
          f"{SEASON_LABELS[args.season]}: {len(played)} played matches "
          f"({played['date'].min().date()} .. {played['date'].max().date()})")
    print(f"Training: {len(prior)} prior seasons ({len(train)} matches), "
          f"seed {args.seed}, starting bankroll ${args.bankroll:,.0f}")
    print("=" * 96)

    agents = {
        "ML-only": BettingAgent(train, bankroll=args.bankroll, seed=args.seed),
        "AI+ML": AIMLBettingAgent(train, bankroll=args.bankroll,
                                  seed=args.seed),
    }
    ledger = {}
    for name, agent in agents.items():
        print(f"\n### {name} — every played match "
              f"(score | pick p | odds | stake | verdict -> result, P/L):")
        header = (f"  {'date':<11}{'match':<30}{'score':>6}{'pick':<10}"
                  f"{'p':>5}{'odds':>6}{'stake':>9}  verdict / result")
        print(header)
        print("  " + "-" * (len(header) - 2))
        rows = []
        for _, m in played.iterrows():
            bank_before = agent.bankroll
            d = agent.decide(m, m["date"], agent.bankroll)
            agent.settle(d, m, agent.bankroll)
            pick, odds, stake = d["decision"], None, d["stake"]
            if pick:
                odds = d["odds"][pick]
            res = RESULT_TO_OUTCOME[m["result"]]
            score = f"{int(m['home_goals'])}-{int(m['away_goals'])}"
            p_txt = f"{d['probs'][pick]:.2f}" if pick else ""
            odds_txt = f"{odds:.2f}" if odds else ""
            stake_txt = f"{stake:,.0f}" if stake else ""
            if pick is None:
                verdict = d["reason"].replace("AI VETO: ", "VETO ")
                verdict = verdict.split("|")[-1].strip()[:46]
                pl, won = 0.0, False
            else:
                won = res == pick
                pl = stake * (odds - 1) if won else -stake
                verdict = (f"{'WIN ' if won else 'LOSS'} — "
                           f"{d['reason'][:42]}")
            print(f"  {m['date'].strftime('%Y-%m-%d'):<11}"
                  f"{m['home_team'][:13] + ' v ' + m['away_team'][:13]:<30}"
                  f"{score:>6}{(pick or '-'):<10}{p_txt:>5}{odds_txt:>6}"
                  f"{stake_txt:>9}  {verdict}")
            rows.append({"date": m["date"], "match": d["match"], "score": score,
                         "pick": pick, "p_model": d["probs"][pick] if pick else None,
                         "stake": stake, "odds": odds, "won": won, "pl": pl,
                         "bankroll_after": agent.bankroll, "result": res,
                         "reason": d["reason"], "bucket": bucket(d["reason"], pick)})
            agent.reveal_result(m)
        ledger[name] = rows

        # ---------------- ANSWERS block ----------------
        df = pd.DataFrame(rows)
        bets = df[df["pick"].notna()]
        staked = bets["stake"].sum()
        pl = df["pl"].sum()
        print(f"\n  >>> {name} — ANSWERS:")
        print(f"      matches bet:            {len(bets)} of {len(df)}")
        print(f"      total staked:           ${staked:,.0f} "
              f"({100 * staked / args.bankroll:.2f}% of starting bankroll)")
        print(f"      bets won / lost:        "
              f"{int(bets['won'].sum())} / {int((bets['pick'].notna() & ~bets['won']).sum())}")
        print(f"      actual money P/L:       ${pl:+,.0f}")
        print(f"      bankroll:               ${args.bankroll:,.0f} -> "
              f"${agent.bankroll:,.0f} ({100 * pl / args.bankroll:+.2f}%)")
        unit_roi = 100 * pl / staked if staked else 0.0
        print(f"      unit ROI (P/L/staked):  {unit_roi:+.1f}%")
        if len(bets):
            print("      the bets:")
            for _, b in bets.iterrows():
                print(f"        {b['date'].date()} {b['match']}: "
                      f"{b['pick']} @{b['odds']:.2f} (model {b['p_model']:.0%}) "
                      f"stake ${b['stake']:,.0f} | {b['reason'][:70]} | "
                      f"score {b['score']} -> {'WIN' if b['won'] else 'LOSS'} "
                      f"{b['pl']:+,.0f}")
        hist = df["bucket"].value_counts()
        print("      why the other matches were skipped:")
        for k, v in hist.items():
            if k != "BET PLACED":
                print(f"        {k:<24} {v} matches")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"season_to_date_{args.league}_{args.season}.csv"
    pd.concat([pd.DataFrame(v).assign(agent=k) for k, v in ledger.items()]
              ).to_csv(out, index=False)
    print(f"\nSaved: {out}")


SEASON_LABELS = {"2021": "2020/21", "2122": "2021/22", "2223": "2022/23",
                 "2324": "2023/24", "2425": "2024/25", "2526": "2025/26",
                 "2627": "2026/27"}

if __name__ == "__main__":
    main()
