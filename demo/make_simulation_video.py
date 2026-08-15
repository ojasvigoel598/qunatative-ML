#!/usr/bin/env python3
"""
Render LIVE videos of the ML betting simulations.

Engine
------
`render_events_video(events, ...)` turns a chronological list of per-match
events (the same schema produced by the synthetic $1M simulation and the real
Serie A / La Liga walk-forward replays) into a 1600x900 mp4:

  * the bankroll equity curve growing match by match,
  * every bet as a marker (green = win, red = loss, size proportional to stake),
  * an "ML THINKING" panel with the model's probability bars, bookmaker odds,
    edge, stake and decision for the current match,
  * PASS decisions when no edge is found,
  * a highlighted "BIGGEST WIN" moment,
  * a fast-forwarded speedrun through the middle,
  * intro and outro cards.

Event schema
------------
{
  "match": "Inter vs Juventus", "probs": {home_win, draw, away_win},
  "odds": {home_win, draw, away_win}, "p_true": {...}, "bankroll": float,
  "is_bet": bool, "best": outcome|None, "edge": float, "stake": float,
  "win": bool|None, "profit": float, "bankroll_after": float,
  "n_bets_so_far": int,
}

Scripts that use it:
  * demo/make_simulation_video.py  (synthetic $1M simulation)
  * demo/make_serie_a_video.py     (real Serie A 2025/26 walk-forward)

Output: demo/output/simulation_live_<policy>.mp4  /  serie_a_live.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402
from demo.simulation import _forward_match_stream, _prediction_table, _world_strengths  # noqa: E402

VIDEO_OUT = Path(__file__).resolve().parent / "output"
VIDEO_OUT.mkdir(parents=True, exist_ok=True)

INITIAL_INVESTMENT = 1_000_000.0
FLAT_STAKE = 10_000.0
EDGE_THRESHOLD = pipeline.EDGE_THRESHOLD
MIN_ODDS = pipeline.MIN_ODDS
MIN_MODEL_PROB = pipeline.MIN_MODEL_PROB
MIN_STAKE = pipeline.MIN_STAKE

FPS = 24
W, H = 16, 9  # figure size (inches) -> 1600x900 px at dpi=100

GREEN = "#06D6A0"
RED = "#EF476F"
BLUE = "#2E86AB"
AMBER = "#F18F01"
DARK = "#101623"
PANEL = "#1B2536"


# ---------------------------------------------------------------------------
# 1. Record one forward trial with full per-match detail (matches simulation.py)
# ---------------------------------------------------------------------------
def record_trial(poisson, ml, rl_agent, strengths, rng, n_matches, policy):
    """Replay the same forward stream as simulation.run_trial, recording events.

    Returns (events, final_bankroll).  Every match is an event; bets carry the
    model's full "thinking" (probs, odds, edge, stake, outcome).
    """
    events = []
    bankroll = INITIAL_INVESTMENT
    n_bets = 0
    prob_table = _prediction_table(poisson, ml)

    for home, away, p_true, opening, closing in _forward_match_stream(strengths, rng, n_matches):
        probs = prob_table[(home, away)]
        edges = poisson.calculate_edge(probs, opening, threshold=EDGE_THRESHOLD)
        best = edges.get("best_value")
        edge = edges.get("max_edge", 0.0)
        ev = {
            "match": f"{home} vs {away}",
            "probs": probs,
            "odds": opening,
            "p_true": p_true,
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
        if best and edge >= EDGE_THRESHOLD:
            odds = opening[best]
            if odds >= MIN_ODDS and probs[best] >= MIN_MODEL_PROB:
                if policy == "flat":
                    stake = min(FLAT_STAKE, bankroll)
                else:
                    if rl_agent is not None:
                        frac = rl_agent.get_stake_fraction(edge, odds, bankroll, INITIAL_INVESTMENT)
                        if frac <= 0:
                            frac = pipeline._fractional_kelly(edge, odds)
                    else:
                        frac = pipeline._fractional_kelly(edge, odds)
                    stake = bankroll * frac
                if stake >= MIN_STAKE:
                    p_win = p_true[best]
                    win = bool(rng.random() < p_win)
                    profit = stake * (odds - 1.0) if win else -stake
                    bankroll += profit
                    ev.update({
                        "is_bet": True,
                        "best": best,
                        "edge": edge,
                        "stake": stake,
                        "win": win,
                        "profit": profit,
                        "bankroll_after": bankroll,
                        "n_bets_so_far": n_bets,
                    })
                    n_bets += 1
        events.append(ev)
    return events, bankroll


# ---------------------------------------------------------------------------
# 2. Frame schedule
# ---------------------------------------------------------------------------
def build_schedule(events, n_bets, biggest_win_idx, detail_bets=36,
                   detail_hold=4, speed_step=8, speed_hold=5):
    """Return a list of (mode, event_idx) frames.

    mode: 'intro' | 'detail' | 'speed' | 'star' | 'outro'
    """
    n_events = len(events)
    sched = [("intro", 0)] * 64

    # Detail phase: first `detail_bets` bets, one event per frame held a few times
    detail_frames = []
    bet_count = 0
    for i in range(n_events):
        if events[i]["is_bet"]:
            bet_count += 1
        if bet_count > detail_bets:
            break
        detail_frames.append(i)
    # Expand with hold
    for i in detail_frames:
        sched.extend([("detail", i)] * detail_hold)

    # Star moment: biggest win
    sched.extend([("star", biggest_win_idx)] * 34)

    # Speedrun phase: skip ahead through the rest
    speed_events = [i for i in range(len(events)) if i > detail_frames[-1]]
    step_events = speed_events[::speed_step]
    for i in step_events:
        sched.extend([("speed", i)] * speed_hold)
    # Always land on the final event so the curve reaches the end
    if speed_events and sched[-1][1] != speed_events[-1]:
        sched.extend([("speed", speed_events[-1])] * speed_hold)

    sched.extend([("outro", n_events - 1)] * 140)
    return sched


# ---------------------------------------------------------------------------
# 3. Frame renderers (module-level so all demo scripts share them)
# ---------------------------------------------------------------------------
def make_figure():
    fig = plt.figure(figsize=(W, H), facecolor=DARK)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.02,
                          left=0.02, right=0.98, top=0.86, bottom=0.06)
    ax_curve = fig.add_subplot(gs[0, 0], facecolor="#0E1420")
    ax_panel = fig.add_subplot(gs[0, 1], facecolor=PANEL)
    ax_header = fig.add_axes([0.02, 0.90, 0.96, 0.08], facecolor=DARK)
    ax_header.axis("off")
    return fig, ax_curve, ax_panel, ax_header


def draw_header(ax, bankroll, n_bets, wins, losses, roi_pct, speedrun):
    ax.clear()
    speed_txt = "   ⏩ SPEEDRUN x8" if speedrun else ""
    ax.text(0.0, 0.55, "ML AGENT  ·  LIVE SIMULATION", color="white",
            fontsize=20, fontweight="bold", va="center", family="DejaVu Sans")
    ax.text(0.0, 0.08, f"Bankroll ${bankroll:,.0f}   ·   Bets {n_bets}   ·   "
                       f"W/L {wins}/{losses}   ·   ROI {roi_pct:+.1f}%{speed_txt}",
            color="#9FB0C3", fontsize=13, va="center", family="DejaVu Sans Mono")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_panel(ax, ev, flash=None):
    """ev: event dict.  flash: 'win' | 'loss' | None"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.06, 0.965, "ML THINKING", color="#7FD1FF", fontsize=15,
            fontweight="bold", va="top")
    ax.text(0.06, 0.90, ev["match"], color="white", fontsize=15,
            fontweight="bold", va="top")

    # Probability bars
    outcomes = [("home_win", "H"), ("draw", "D"), ("away_win", "A")]
    y0 = 0.62
    for i, (k, lbl) in enumerate(outcomes):
        p = ev["probs"][k]
        y = y0 - i * 0.115
        chosen = (k == ev["best"])
        bar_c = "#3D5A80" if not chosen else BLUE
        ax.add_patch(FancyBboxPatch((0.06, y), 0.06, 0.065,
                                    boxstyle="round,pad=0.004", fc=bar_c, ec="none"))
        ax.add_patch(FancyBboxPatch((0.12, y), 0.62 * p, 0.065,
                                    boxstyle="round,pad=0.004", fc=bar_c, ec="none"))
        ax.text(0.78, y + 0.015, f"{p*100:.1f}%", color="white", fontsize=12,
                family="DejaVu Sans Mono", va="center")
        ax.text(0.07, y + 0.015, lbl, color="white", fontsize=11,
                fontweight="bold", va="center")
        if chosen:
            ax.text(0.93, y + 0.015, "◀ BET", color=AMBER, fontsize=11,
                    fontweight="bold", va="center")

    # Odds line
    ax.text(0.06, 0.22, "Odds   " + "  ".join(
        f"{ev['odds'][k]:.2f}" for k, _ in outcomes), color="#C8D3E0",
        fontsize=11, family="DejaVu Sans Mono")
    # Edge + stake
    if ev["is_bet"]:
        ax.text(0.06, 0.13, f"Edge +{ev['edge']*100:.1f}%   Stake ${ev['stake']:,.0f}",
                color=AMBER, fontsize=13, fontweight="bold",
                family="DejaVu Sans Mono")
        # Outcome flash
        if flash == "win":
            ax.text(0.5, 0.45, f"WIN  +${ev['profit']:,.0f}", color=GREEN,
                    fontsize=24, fontweight="bold", ha="center", va="center")
        elif flash == "loss":
            ax.text(0.5, 0.45, f"LOSS  ${ev['profit']:,.0f}", color=RED,
                    fontsize=24, fontweight="bold", ha="center", va="center")
    else:
        ax.text(0.06, 0.13, "PASS — no edge ≥ 3%", color="#8B9BB0",
                fontsize=13, family="DejaVu Sans Mono")
    # Optional dynamic-thinking trace (set by make_dynamic_video.py)
    if ev.get("thinking"):
        ax.text(0.06, 0.055, ev["thinking"], color="#7FD1FF", fontsize=10.5,
                family="DejaVu Sans Mono", va="bottom")


def draw_curve(ax, events, upto, biggest_win_idx):
    ax.clear()
    ax.set_facecolor("#0E1420")
    bet_idx = [e["n_bets_so_far"] for e in events[:upto] if e["is_bet"]]
    bank = [e["bankroll_after"] for e in events[:upto] if e["is_bet"]]
    if not bet_idx:
        bet_idx, bank = [0], [INITIAL_INVESTMENT]
    ax.plot(bet_idx, bank, color="#E8EDF2", linewidth=1.8, zorder=2)
    # markers (guard: no bets placed yet in early frames)
    stakes = np.array([e["stake"] for e in events[:upto] if e["is_bet"]])
    wins = np.array([e["win"] for e in events[:upto] if e["is_bet"]])
    if len(stakes) > 0:
        colors = np.where(wins, GREEN, RED)
        sizes = 24 + 90 * (stakes / max(stakes.max(), 1.0))
        ax.scatter(bet_idx, bank, s=sizes, c=colors, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
    # start line
    ax.axhline(INITIAL_INVESTMENT, color="#44506A", linestyle="--", linewidth=1.2)
    ax.text(len(bet_idx) * 0.985, INITIAL_INVESTMENT * 1.01, "$1M start",
            color="#6B7A93", fontsize=10, ha="right")
    # biggest win star
    if biggest_win_idx is not None and biggest_win_idx < upto:
        ev = events[biggest_win_idx]
        ax.scatter([ev["n_bets_so_far"]], [ev["bankroll_after"]], s=260, c="none",
                   edgecolors=AMBER, linewidths=2.2, zorder=5)
        ax.annotate("BIGGEST WIN", (ev["n_bets_so_far"], ev["bankroll_after"]),
                    xytext=(10, 14), textcoords="offset points", color=AMBER,
                    fontsize=11, fontweight="bold", zorder=6)
    ax.set_title("Bankroll  ($M)  vs  bet number", color="#C8D3E0", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Bet number", color="#C8D3E0", fontsize=11)
    ax.set_ylabel("Bankroll ($M)", color="#C8D3E0", fontsize=11)
    ax.set_xlim(-2, max([e["n_bets_so_far"] for e in events if e["is_bet"]]) + 4)
    ax.set_ylim(min(bank) * 0.98, max(bank) * 1.02)
    ax.tick_params(colors="#9FB0C3")
    for s in ax.spines.values():
        s.set_color("#2A3547")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v/1e6:.1f}")


def draw_card(fig, ax, title, lines, accent=GREEN, footer=None):
    """Draw an intro/outro card.  NOTE: does NOT call fig.clear() — clearing the
    figure would destroy the shared subplot axes (ax_header/ax_curve/ax_panel)
    so every later frame would draw onto detached axes and render blank.
    The caller passes a dedicated full-figure card axes instead."""
    ax.clear()
    ax.axis("off")
    ax.set_facecolor(DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.82, title, color="white", fontsize=34, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.plot([0.3, 0.7], [0.70, 0.70], color=accent, linewidth=3,
            transform=ax.transAxes)
    for i, ln in enumerate(lines):
        ax.text(0.5, 0.60 - i * 0.075, ln, color="#DCE4EE", fontsize=18,
                ha="center", va="center", family="DejaVu Sans Mono",
                transform=ax.transAxes)
    if footer:
        ax.text(0.5, 0.05, footer, color="#8B9BB0", fontsize=12, ha="center",
                va="center", transform=ax.transAxes)
    return ax


# ---------------------------------------------------------------------------
# 4. Shared video engine
# ---------------------------------------------------------------------------
def render_events_video(events, out_path, title="ML AGENT  ·  LIVE SIMULATION",
                        intro_lines=None, outro_lines=None, footer=None,
                        biggest_win_idx=None, poster_path=None):
    """Render `events` to an mp4.  All demo scripts funnel through this."""
    n_bets = sum(1 for e in events if e["is_bet"])
    if biggest_win_idx is None:
        bets = [e for e in events if e["is_bet"]]
        if bets:
            biggest = max(bets, key=lambda e: e["profit"])
            biggest_win_idx = events.index(biggest)

    sched = build_schedule(events, n_bets, biggest_win_idx)
    print(f"  Frames: {len(sched)}  (~{len(sched)/FPS:.0f}s at {FPS}fps)")

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    fig, ax_curve, ax_panel, ax_header = make_figure()
    writer = animation.FFMpegWriter(fps=FPS, codec="libx264", bitrate=5000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    flash_for = 0
    last_flash = None
    card_ax = None  # dedicated full-figure axes for intro/outro cards

    def frame_fn(fr):
        nonlocal flash_for, last_flash, card_ax
        mode, ev_idx = sched[fr]
        if mode in ("intro", "outro"):
            # card mode: hide the shared axes, (re)create a full-figure card axes
            if card_ax is None:
                for a in (ax_curve, ax_panel, ax_header):
                    a.set_visible(False)
                card_ax = fig.add_axes([0.06, 0.10, 0.88, 0.80])
            if mode == "intro":
                draw_card(fig, card_ax, title, intro_lines or [
                    "PoissonElo  +  Gradient Boosting  +  Q-Learning",
                    "forward simulation · point-in-time knowledge",
                    "watch the model think, bet, win and lose ..."],
                    accent=GREEN, footer=footer or
                    "synthetic calibrated world · not a prediction of real returns")
            else:
                draw_card(fig, card_ax, "FINISHED", outro_lines or [
                    f"Final bankroll   ${events[-1]['bankroll_after']:,.0f}"],
                    accent=GREEN, footer=footer or "this path is ONE draw - variance is the story")
            return []
        # live mode: leave card mode (restore shared axes, drop card axes)
        if card_ax is not None:
            card_ax.remove()
            card_ax = None
            for a in (ax_curve, ax_panel, ax_header):
                a.set_visible(True)
        if mode == "star":
            ev_idx = biggest_win_idx
        speedrun = mode == "speed"
        ev = events[ev_idx]
        if ev["is_bet"] and (mode in ("detail", "star") or
                             (mode == "speed" and fr > 0 and sched[fr-1][0] != "speed")):
            last_flash = "win" if ev["win"] else "loss"
            flash_for = 6
        if flash_for > 0:
            flash_for -= 1
        else:
            last_flash = None

        wins_so_far = sum(1 for e in events[:ev_idx+1] if e["is_bet"] and e["win"])
        losses_so_far = sum(1 for e in events[:ev_idx+1] if e["is_bet"] and not e["win"])
        bets_so_far = wins_so_far + losses_so_far
        bankroll_so_far = events[ev_idx]["bankroll_after"] if ev["is_bet"] else events[ev_idx]["bankroll"]
        roi_so_far = (bankroll_so_far - INITIAL_INVESTMENT) / INITIAL_INVESTMENT * 100
        draw_header(ax_header, bankroll_so_far, bets_so_far, wins_so_far,
                    losses_so_far, roi_so_far, speedrun)
        draw_curve(ax_curve, events, ev_idx + 1, biggest_win_idx if mode != "star" else None)
        draw_panel(ax_panel, ev, flash=last_flash)
        return []

    with writer.saving(fig, str(out_path), dpi=100):
        for fr in range(len(sched)):
            frame_fn(fr)
            writer.grab_frame()
    plt.close(fig)
    print(f"  [OK] Video saved: {out_path}")

    # poster frame (mid-simulation)
    if poster_path:
        fig2, axc, axp, axh = make_figure()
        mid = sched[len(sched)//2][1]
        ev = events[mid]
        draw_header(axh, events[mid]["bankroll"], events[mid]["n_bets_so_far"],
                    sum(1 for e in events[:mid+1] if e["is_bet"] and e["win"]),
                    sum(1 for e in events[:mid+1] if e["is_bet"] and not e["win"]),
                    (events[mid]["bankroll"] - INITIAL_INVESTMENT)/INITIAL_INVESTMENT*100,
                    False)
        draw_curve(axc, events, mid + 1, biggest_win_idx)
        draw_panel(axp, ev, flash="win" if ev["is_bet"] and ev["win"] else None)
        fig2.savefig(poster_path, dpi=100)
        plt.close(fig2)
        print(f"  [OK] Poster saved: {poster_path}")


# ---------------------------------------------------------------------------
# 5. Synthetic $1M simulation entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Render the $1M simulation live video")
    parser.add_argument("--matches", type=int, default=1200)
    parser.add_argument("--policy", choices=["flat", "kelly"], default="flat")
    parser.add_argument("--trial", type=int, default=0, help="trial index (rng seed 1000+trial)")
    args = parser.parse_args()

    print("=" * 70)
    print("RENDERING $1M SIMULATION VIDEO (live replay)")
    print(f"  policy={args.policy}  trial={args.trial}  matches={args.matches}")
    print("=" * 70)

    # --- train the same models as the simulation ---
    df, _ = pipeline.load_or_generate_data(n_matches=args.matches, seed=42)
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: int(n * 0.65)].copy()
    valid_df = df.iloc[int(n * 0.65): int(n * 0.80)].copy()
    poisson, ml = pipeline.train_models(train_df, use_ml=True, verbose=True)

    rl_agent = None
    if args.policy == "kelly":
        from models.rl_staking_agent import QLearningStakingAgent
        experiences = pipeline._discovery_experiences(valid_df, poisson, ml, 1000.0)
        rl_agent = QLearningStakingAgent()
        rl_agent.train(experiences, episodes=200)

    # --- record one trial ---
    strengths = _world_strengths(42)
    rng = np.random.default_rng(1000 + args.trial)
    events, final_bankroll = record_trial(poisson, ml, rl_agent, strengths, rng,
                                          args.matches, args.policy)
    n_bets = sum(1 for e in events if e["is_bet"])
    wins = sum(1 for e in events if e["is_bet"] and e["win"])
    losses = n_bets - wins
    roi = (final_bankroll - INITIAL_INVESTMENT) / INITIAL_INVESTMENT * 100
    print(f"  Final bankroll: ${final_bankroll:,.0f}  ROI {roi:+.1f}%  "
          f"bets {n_bets} (W {wins} / L {losses})")

    out = VIDEO_OUT / f"simulation_live_{args.policy}.mp4"
    poster = VIDEO_OUT / "simulation_live_poster.png"
    render_events_video(
        events, out,
        title="ML AGENT  ·  $1,000,000 LIVE SIMULATION",
        intro_lines=[
            "IF I INVESTED $1,000,000 ...",
            "",
            "PoissonElo  +  Gradient Boosting  +  Q-Learning",
            f"forward simulation: {args.matches} matches ~ 3.3 years",
            "policy: " + ("flat $10K / bet" if args.policy == "flat"
                          else "quarter-Kelly / RL"),
            "",
            "watch the model think, bet, win and lose ...",
        ],
        outro_lines=[
            f"Final bankroll   ${final_bankroll:,.0f}",
            f"ROI {roi:+.1f}%   ·   {wins} W / {losses} L",
            f"Bets {n_bets}   ·   biggest win +${max((e['profit'] for e in events if e['is_bet']), default=0):,.0f}",
            "25-trial distribution: mean $1.44M · median $1.38M",
            "P(end in profit) 88%  ·  worst path $802K",
        ],
        footer="this path is ONE Monte-Carlo draw - variance is the story",
        poster_path=poster)


if __name__ == "__main__":
    main()
