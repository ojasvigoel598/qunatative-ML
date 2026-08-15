#!/usr/bin/env python3
"""
Render a LIVE video of the REAL Serie A 2025/26 walk-forward replay.

The model (adaptive: PoissonElo + Gradient Boosting + online adaptation) is
trained on REAL Serie A 2020/21-2023/24, then walks REAL Serie A 2025/26
match-by-match with point-in-time knowledge only:

  * pre-match Elo/form built from matches already played,
  * the real B365 opening odds from the fixture sheet,
  * quarter-Kelly capped at 5% of bankroll, 15% daily cap, survival mode
    (flat 0.5% stakes) if the bankroll ever drops below $100K,
  * the real result is revealed after each match (bankroll + online state
    advance; no future information is ever used).

Output: demo/output/serie_a_live.mp4 (+ serie_a_live_poster.png)

Usage:
    python demo/make_serie_a_video.py           # downloads real data if needed
    python demo/make_serie_a_video.py --offline
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.real_data import get_season  # noqa: E402
from models.adaptive_model import AdaptiveMatchPredictor, CLASS_MAP  # noqa: E402
from demo.make_simulation_video import (  # noqa: E402
    VIDEO_OUT, INITIAL_INVESTMENT, render_events_video,
)

TRAIN_SEASONS = ["2021", "2122", "2223", "2324"]   # 2020/21 .. 2023/24
TEST_SEASON = "2526"

EDGE_THRESHOLD = 0.03
PROB_FLOOR = 0.40
KELLY_FRACTION = 0.25
STAKE_CAP = 0.05
DAILY_CAP = 0.15
SURVIVAL_FLOOR = 0.10
SURVIVAL_EDGE = 0.01
SURVIVAL_STAKE = 0.005
MIN_STAKE = 100.0
RESULT_KEY = {"H": "home_win", "D": "draw", "A": "away_win"}


def replay_and_record(train: pd.DataFrame, season: pd.DataFrame) -> tuple:
    """Walk Serie A 25/26 point-in-time with the adaptive model.

    Returns (events, final_bankroll).  Same event schema as the synthetic
    simulation so the shared video engine can render it.
    """
    model = AdaptiveMatchPredictor(static=False)
    model.train(train)

    bankroll = INITIAL_INVESTMENT
    n_bets = 0
    events = []
    daily_used, last_day = 0.0, None

    for _, r in season.iterrows():
        match_date = r["date"]
        if last_day is not None and match_date != last_day:
            daily_used = 0.0
        last_day = match_date

        probs = model.predict(r["home_team"], r["away_team"])
        odds = {"home_win": r.get("odds_home"), "draw": r.get("odds_draw"),
                "away_win": r.get("odds_away")}
        edges = {k: (probs[k] * odds[k] - 1.0) for k in odds
                 if pd.notna(odds[k]) and odds[k] > 1.0}

        ev = {
            "match": f"{r['home_team']} vs {r['away_team']}",
            "probs": probs,
            "odds": odds,
            "bankroll": bankroll,
            "is_bet": False,
            "best": None,
            "edge": 0.0,
            "stake": 0.0,
            "win": None,
            "profit": 0.0,
            "bankroll_after": bankroll,
            "n_bets_so_far": n_bets,
        }

        placed = None
        if edges:
            best = max(edges, key=edges.get)
            mode = "survival" if bankroll < SURVIVAL_FLOOR * INITIAL_INVESTMENT else "aggressive"
            threshold = SURVIVAL_EDGE if mode == "survival" else EDGE_THRESHOLD
            prob_ok = (mode == "survival") or (probs[best] >= PROB_FLOOR)
            if edges[best] > threshold and prob_ok:
                odds_bet = odds[best]
                if mode == "survival":
                    stake = SURVIVAL_STAKE * bankroll
                else:
                    f = max(0.0, (probs[best] * odds_bet - 1.0) / (odds_bet - 1.0))
                    stake = min(KELLY_FRACTION * f * bankroll, STAKE_CAP * bankroll)
                remaining_daily = max(0.0, DAILY_CAP * bankroll - daily_used)
                stake = min(stake, remaining_daily)
                if stake >= MIN_STAKE:
                    daily_used += stake
                    placed = (best, stake, odds_bet, probs[best], edges[best], mode)

        won = placed is not None and RESULT_KEY[r["result"]] == placed[0]
        if placed is not None:
            outcome, stake, odds_bet, prob, edge, mode = placed
            profit = (stake * (odds_bet - 1.0)) if won else (-stake)
            bankroll += profit
            ev.update({
                "is_bet": True,
                "best": outcome,
                "edge": edge,
                "stake": stake,
                "win": bool(won),
                "profit": profit,
                "bankroll_after": bankroll,
                "n_bets_so_far": n_bets,
            })
            n_bets += 1

        # reveal the result to the adaptive model (online Elo/form + refits)
        model.observe(r["home_team"], r["away_team"],
                      float(r["home_goals"]), float(r["away_goals"]),
                      r["result"])
        events.append(ev)

    return events, bankroll


def main():
    parser = argparse.ArgumentParser(description="Real Serie A 25/26 live replay video")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("RENDERING REAL SERIE A 2025/26 LIVE REPLAY VIDEO (adaptive model)")
    print("=" * 70)

    train = pd.concat([get_season("I1", s, args.offline) for s in TRAIN_SEASONS],
                      ignore_index=True).sort_values("date").reset_index(drop=True)
    season = get_season("I1", TEST_SEASON, args.offline)
    print(f"  Trained on {len(train)} real Serie A matches (2020/21-2023/24)")
    print(f"  Replaying {len(season)} real Serie A matches (2025/26), $1M start")

    events, final_bankroll = replay_and_record(train, season)
    n_bets = sum(1 for e in events if e["is_bet"])
    wins = sum(1 for e in events if e["is_bet"] and e["win"])
    losses = n_bets - wins
    roi = (final_bankroll - INITIAL_INVESTMENT) / INITIAL_INVESTMENT * 100
    print(f"  Final bankroll: ${final_bankroll:,.0f}  ROI {roi:+.1f}%  "
          f"bets {n_bets} (W {wins} / L {losses})")

    out = VIDEO_OUT / "serie_a_live.mp4"
    poster = VIDEO_OUT / "serie_a_live_poster.png"
    render_events_video(
        events, out,
        title="ML AGENT  ·  SERIE A 2025/26  ·  $1M LIVE REPLAY",
        intro_lines=[
            "REAL SERIE A 2025/26 — POINT-IN-TIME REPLAY",
            "",
            "adaptive model: PoissonElo + Gradient Boosting",
            f"trained on {len(train)} real Serie A matches (20/21-23/24)",
            "real B365 odds · quarter-Kelly · 15% daily cap",
            "survival mode below $100K",
            "",
            "watch it think, bet, win and lose ...",
        ],
        outro_lines=[
            f"Final bankroll   ${final_bankroll:,.0f}",
            f"ROI {roi:+.1f}%   ·   {wins} W / {losses} L",
            f"Bets {n_bets}   ·   biggest win +${max((e['profit'] for e in events if e['is_bet']), default=0):,.0f}",
            "real results, revealed chronologically",
            "no future information used",
        ],
        footer="real-match replay · one season · small sample",
        poster_path=poster)


if __name__ == "__main__":
    main()
