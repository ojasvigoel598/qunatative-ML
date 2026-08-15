#!/usr/bin/env python3
"""
Render a LIVE video of the $1M simulation driven by the DYNAMIC THINKING LAYER.

Shows, in real time, the adaptive decision process itself:

  * the bankroll equity curve growing match by match,
  * every bet as a marker (green = win, red = loss, size proportional to stake),
  * an "ML THINKING" panel with the model's probability bars, the *sharp line*
    implied probabilities, the public-vs-sharp market split, the adaptive
    model-vs-market weight, the edge, the uncertainty shrink, the drawdown
    risk factor and the stake — i.e. exactly what the layer was thinking,
  * PASS decisions when no edge is found,
  * a BIGGEST-WIN highlight and a speedrun through the middle,
  * intro and outro cards.

Output: demo/output/simulation_live_dynamic.mp4 (+ poster)

Run:
    python demo/make_dynamic_video.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402
from models.dynamic_thinking import DynamicThinkingLayer  # noqa: E402
from demo.simulation import (  # noqa: E402
    INITIAL_INVESTMENT, _forward_match_stream, _world_strengths,
)
from demo.make_simulation_video import render_events_video, VIDEO_OUT  # noqa: E402


def record_dynamic_trial(train_df, strengths, rng, n_matches):
    """Walk one forward trial through the FULLY dynamic thinking layer
    (self-refitting base + multi-signal fusion), recording the decision trace."""
    layer = DynamicThinkingLayer(train_df=train_df, bankroll=INITIAL_INVESTMENT,
                                 seed=int(rng.integers(0, 1_000_000)))
    events = []
    bankroll = INITIAL_INVESTMENT
    n_bets = 0

    for day, (home, away, p_true, opening, closing) in enumerate(
            _forward_match_stream(strengths, rng, n_matches)):
        extra_book = {k: closing[k] * float(rng.uniform(0.98, 1.02)) for k in closing}
        conditions = {"away_win": float(rng.uniform(-0.02, 0.02))} if day % 23 == 0 else None
        decision = layer.think(home, away, opening, closing,
                               extra_books=[extra_book], conditions=conditions,
                               current_day=day)
        outcome = decision["decision"]

        # sample the TRUE outcome for this match (the world, not the model)
        roll = rng.random()
        cum = np.cumsum([p_true["home_win"], p_true["draw"], p_true["away_win"]])
        result = "H" if roll < cum[0] else ("D" if roll < cum[1] else "A")
        hg = 2 if result == "H" else (1 if result == "D" else 0)
        ag = 1 if result == "H" else (1 if result == "D" else 2)

        thinking_txt = ("mkt_w {:.2f}  conf {:.2f}  disp {:.2f}  rest {}/{}d".format(
            decision["market_weight"], decision["confidence"],
            decision["dispersion"], decision["rest_days"][0],
            decision["rest_days"][1])) + decision["conditions"]
        ev = {
            "match": f"{home} vs {away}",
            "probs": {"home_win": decision["fused"][0],
                      "draw": decision["fused"][1],
                      "away_win": decision["fused"][2]},
            "odds": opening,
            "bankroll": bankroll,
            "is_bet": False,
            "best": None,
            "edge": 0.0,
            "stake": 0.0,
            "win": None,
            "profit": 0.0,
            "bankroll_after": bankroll,
            "n_bets_so_far": n_bets,
            "thinking": thinking_txt,
        }
        if outcome is not None:
            won = (result == "H" and outcome == "home_win") or \
                  (result == "D" and outcome == "draw") or \
                  (result == "A" and outcome == "away_win")
            stake = decision["stake"]
            profit = stake * (opening[outcome] - 1.0) if won else -stake
            bankroll += profit
            ev.update({
                "is_bet": True,
                "best": outcome,
                "edge": decision["edge"],
                "stake": stake,
                "win": won,
                "profit": profit,
                "bankroll_after": bankroll,
                "n_bets_so_far": n_bets,
            })
            n_bets += 1
        layer.observe(home, away, hg, ag, result, decision, opening,
                      current_day=day)
        events.append(ev)
    return events, bankroll


def main():
    print("=" * 70)
    print("RENDERING $1M DYNAMIC-THINKING SIMULATION VIDEO")
    print("=" * 70)

    df, _ = pipeline.load_or_generate_data(n_matches=1200, seed=42)
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: int(n * 0.65)].copy()

    strengths = _world_strengths(42)
    rng = np.random.default_rng(1000 + 0)   # same trial 0 as the other videos
    events, final_bankroll = record_dynamic_trial(train_df, strengths, rng, 1200)

    n_bets = sum(1 for e in events if e["is_bet"])
    wins = sum(1 for e in events if e["is_bet"] and e["win"])
    losses = n_bets - wins
    roi = (final_bankroll - INITIAL_INVESTMENT) / INITIAL_INVESTMENT * 100
    print(f"  Final bankroll: ${final_bankroll:,.0f}  ROI {roi:+.1f}%  "
          f"bets {n_bets} (W {wins} / L {losses})")

    out = VIDEO_OUT / "simulation_live_dynamic.mp4"
    poster = VIDEO_OUT / "simulation_live_dynamic_poster.png"
    render_events_video(
        events, out,
        title="ML AGENT  ·  DYNAMIC THINKING  ·  $1M LIVE",
        intro_lines=[
            "EVERY DECISION THROUGH THE THINKING LAYER",
            "",
            "model + sharp-line market split + fatigue + conditions",
            "model-vs-market weight adapts from rolling calibration",
            "CONFIDENCE-AWARE: sure picks commit bigger stakes",
            "confidence-gated refits when the model loses its grip",
            "survival mode below $100K",
            "",
            "watch what it thinks, then bet ...",
        ],
        outro_lines=[
            f"Final bankroll   ${final_bankroll:,.0f}",
            f"ROI {roi:+.1f}%   ·   {wins} W / {losses} L",
            f"Bets {n_bets}   ·   biggest win +${max((e['profit'] for e in events if e['is_bet']), default=0):,.0f}",
            "confidence-aware: median $1.21M · 90% [$1.01M .. $1.61M] (12 trials)",
            "13.5 refits/trial (1.5 confidence-gated) · model weight 0.48",
        ],
        footer="one Monte-Carlo draw · synthetic calibrated world",
        poster_path=poster)


if __name__ == "__main__":
    main()
