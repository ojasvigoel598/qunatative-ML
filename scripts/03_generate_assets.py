#!/usr/bin/env python3
"""
Generate the repository assets used by the README:

  assets/architecture.png      - pipeline diagram (matplotlib, no external deps)
  assets/backtest_analysis.png - curated result plots (copied from backtests/results)

Run after the backtest scripts so the plots reflect the latest run:
    python run_full_ml_rl.py
    python scripts/03_generate_assets.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
BACKTEST = ROOT / "backtests" / "results"
ASSETS.mkdir(parents=True, exist_ok=True)

BOX_KW = dict(boxstyle="round,pad=0.55", facecolor="#f2f7fb",
              edgecolor="#2E86AB", linewidth=1.6)
HEAD_KW = dict(boxstyle="round,pad=0.55", facecolor="#2E86AB",
               edgecolor="#1d5f80", linewidth=1.6)


def draw_architecture():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")

    def box(x, y, w, h, text, kw=BOX_KW, fontsize=11, color="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, **kw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=color, family="sans-serif")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.8))

    # Title
    ax.text(5.5, 9.2, "Quantitative Sports Betting Model - Pipeline",
            ha="center", fontsize=16, fontweight="bold", color="#1d3d55")
    ax.text(5.5, 8.7, "Poisson + Elo  |  Gradient Boosting  |  Q-Learning staking",
            ha="center", fontsize=11, color="#555")

    # Column 1: data & features
    box(0.6, 6.6, 2.6, 1.1, "Match data\n(synthetic world or CSV)", HEAD_KW, color="white")
    box(0.6, 4.9, 2.6, 1.1, "Features\nElo ratings + shifted rolling form")
    box(0.6, 3.2, 2.6, 1.1, "Train / validation / test split\n(65 / 15 / 20, chronological)")

    # Column 2: models
    box(4.0, 6.6, 3.0, 1.1, "Layer 1: PoissonElo\nPoisson regression + Elo")
    box(4.0, 4.9, 3.0, 1.1, "Layer 2: Gradient Boosting\ncalibrated probabilities")
    box(4.0, 3.2, 3.0, 1.1, "Hybrid ensemble\np = (p_poisson + p_ml) / 2")

    # Column 3: betting engine
    box(7.8, 6.6, 3.0, 1.1, "Value detection\nedge = p x odds - 1 > 3%, p >= 0.40")
    box(7.8, 4.9, 3.0, 1.1, "Layer 3: Q-Learning staking\nKelly-multiplier agent")
    box(7.8, 3.2, 3.0, 1.1, "Backtest\nresolve bets, track bankroll, CLV")

    # Column 4: output
    box(11.6, 4.9, 2.2, 1.4, "Metrics & plots\nROI, Sharpe, max DD,\nequity curve, CLV", HEAD_KW, color="white")

    # Arrows
    arrow(3.2, 7.15, 4.0, 7.15)          # data -> poisson
    arrow(3.2, 5.45, 4.0, 5.45)          # features -> ml
    arrow(3.2, 3.75, 4.0, 3.75)          # split -> ensemble
    arrow(2.6, 6.0, 2.6, 5.6)            # data -> features (vertical)
    arrow(2.6, 4.3, 2.6, 3.9)            # features -> split
    arrow(7.0, 7.15, 7.8, 7.15)          # poisson -> value
    arrow(7.0, 5.45, 7.8, 5.45)          # ml -> value
    arrow(7.0, 3.75, 7.8, 3.75)          # ensemble -> backtest
    arrow(9.2, 6.6, 9.2, 6.1)            # value -> staking
    arrow(9.2, 4.9, 9.2, 4.35)           # staking -> backtest
    arrow(10.8, 3.75, 11.6, 5.1)         # backtest -> output
    arrow(7.8, 3.2, 5.5, 1.9)            # backtest loops to features (retrain)

    ax.text(6.5, 1.9, "iterate", ha="center", fontsize=9, color="#777")

    plt.tight_layout()
    plt.savefig(ASSETS / "architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {ASSETS / 'architecture.png'}")


def copy_plots():
    """Curate the latest result plots into assets/."""
    for name in ["backtest_analysis_ml_rl.png", "backtest_summary_ml_rl.png"]:
        src = BACKTEST / name
        if src.exists():
            dest = ASSETS / name
            dest.write_bytes(src.read_bytes())
            print(f"[OK] Copied {src.name} -> assets/")
        else:
            print(f"[WARN] {src} not found - run run_full_ml_rl.py first")


if __name__ == "__main__":
    draw_architecture()
    copy_plots()
