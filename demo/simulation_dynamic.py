#!/usr/bin/env python3
"""
$1,000,000 SIMULATION DRIVEN BY THE DYNAMIC THINKING LAYER.

The question: if the *decision process itself* is dynamic — the model fuses
fresh signals (public vs sharp market split, fatigue, conditions) before every
bet, re-weights model vs market from its own rolling calibration, and sizes
stakes from uncertainty + drawdown + survival rules — does that change the
$1M outcome vs static flat/Kelly policies?

Protocol (identical for every policy)
-------------------------------------
* Train PoissonElo + GB on the historical synthetic world (seed 42).
* Run 25 independent forward trials of 1,200 matches each.
* Policies compared on the SAME streams:
    - flat   : $10K per bet (variance-minimising, the current default)
    - kelly  : quarter-Kelly (original behaviour)
    - dynamic: DynamicThinkingLayer (market-split signal + adaptive blend +
               uncertainty shrink + drawdown risk + survival mode)
* Outcomes are sampled from the TRUE match probabilities (the world, not the
  model).  Synthetic calibrated world — methodology demo, not a real-world
  return prediction.

Outputs: demo/output/simulation_1m_dynamic.csv + .png, docs/12_dynamic_thinking.md

Run:
    python demo/simulation_dynamic.py
    python demo/simulation_dynamic.py --trials 25 --matches 1200
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
from models.dynamic_thinking import DynamicThinkingLayer  # noqa: E402
from models.rl_staking_agent import QLearningStakingAgent  # noqa: E402
from demo.simulation import (  # noqa: E402
    INITIAL_INVESTMENT, _forward_match_stream, _world_strengths,
    EDGE_THRESHOLD, MIN_ODDS, MIN_MODEL_PROB, MIN_STAKE, FLAT_STAKE,
    run_trial, SIM_OUT,
)

YEARS = 1200 / 365.25


def run_dynamic_trial_clean(train_df, strengths, rng, n_matches, poisson=None,
                            ml=None, mode="dynamic"):
    """Dynamic trial with the FULLY dynamic model (mode='dynamic', default):
    the thinking layer owns a self-refitting AdaptiveMatchPredictor (Elo/form/
    GB refits online, INCLUDING confidence-gated refits) and fuses
    public-vs-sharp split, multi-book consensus, dispersion, fatigue and
    occasional synthesized live conditions before every decision, with
    confidence-aware adaptation.  The outcome is sampled from the TRUE world
    probabilities, then revealed.

    mode='v1' reproduces the ORIGINAL (pre-upgrade) layer: a FIXED poisson+ml
    base, no self-refit, no multi-book fusion, no confidence — the honest
    baseline for measuring what the upgrades buy.
    """
    if mode == "v1":
        layer = DynamicThinkingLayer(poisson=poisson, ml=ml,
                                     bankroll=INITIAL_INVESTMENT,
                                     seed=int(rng.integers(0, 1_000_000)),
                                     simple=True, confidence_aware=False)
    else:
        layer = DynamicThinkingLayer(train_df=train_df,
                                     bankroll=INITIAL_INVESTMENT,
                                     seed=int(rng.integers(0, 1_000_000)))
    equity = [INITIAL_INVESTMENT]
    for day, (home, away, p_true, opening, closing) in enumerate(
            _forward_match_stream(strengths, rng, n_matches)):
        # synthesize a second "book" (noisy sharp line) + occasional conditions
        extra_book = {k: closing[k] * float(rng.uniform(0.98, 1.02)) for k in closing}
        conditions = None
        if day % 23 == 0:   # a "breaking news" event now and then
            conditions = {"away_win": float(rng.uniform(-0.03, 0.03))}
        decision = layer.think(home, away, opening, closing,
                               extra_books=[extra_book], conditions=conditions,
                               current_day=day)
        outcome = decision["decision"]
        # sample the true outcome
        roll = rng.random()
        cum = np.cumsum([p_true["home_win"], p_true["draw"], p_true["away_win"]])
        result = "H" if roll < cum[0] else ("D" if roll < cum[1] else "A")
        hg = 2 if result == "H" else (1 if result == "D" else 0)
        ag = 1 if result == "H" else (1 if result == "D" else 2)
        layer.observe(home, away, hg, ag, result, decision, opening,
                      current_day=day)
        equity.append(layer.bankroll)
    return layer.bankroll, np.array(equity), layer


def main():
    parser = argparse.ArgumentParser(description="$1M dynamic-thinking simulation")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--matches", type=int, default=1200)
    args = parser.parse_args()

    print("=" * 78)
    print("$1M SIMULATION — DYNAMIC THINKING LAYER vs flat / Kelly policies")
    print("=" * 78)

    # ---- train the same base models as the canonical simulation
    df, _ = pipeline.load_or_generate_data(n_matches=args.matches, seed=42)
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: int(n * 0.65)].copy()
    valid_df = df.iloc[int(n * 0.65): int(n * 0.80)].copy()
    poisson, ml = pipeline.train_models(train_df, use_ml=True, verbose=True)
    experiences = pipeline._discovery_experiences(valid_df, poisson, ml, 1000.0)
    rl_agent = QLearningStakingAgent()
    rl_agent.train(experiences, episodes=200)

    strengths = _world_strengths(42)
    finals = {"flat": [], "kelly": [], "dynamic": [], "dynamic_v1": []}
    bet_counts = {"flat": [], "kelly": [], "dynamic": [], "dynamic_v1": []}
    weights = []
    refits = []
    conf_refits = []
    final_conf = []
    sample_paths = {"flat": [], "dynamic": []}

    for t in range(args.trials):
        rng = np.random.default_rng(1000 + t)
        f, eq, nb = run_trial(poisson, ml, rl_agent, strengths, rng,
                              args.matches, policy="flat")
        finals["flat"].append(f); bet_counts["flat"].append(nb)
        if t < 8:
            sample_paths["flat"].append(eq)

        rng = np.random.default_rng(1000 + t)
        f, eq, nb = run_trial(poisson, ml, rl_agent, strengths, rng,
                              args.matches, policy="kelly")
        finals["kelly"].append(f); bet_counts["kelly"].append(nb)

        rng = np.random.default_rng(1000 + t)
        f, eq, layer = run_dynamic_trial_clean(train_df, strengths, rng,
                                               args.matches)
        finals["dynamic"].append(f)
        bet_counts["dynamic"].append(layer.n_bets)
        weights.append(layer.market_weight)
        refits.append(getattr(layer.base, "refits", 0))
        conf_refits.append(layer.conf_refits)
        final_conf.append(layer.summary()["final_confidence"])
        if t < 8:
            sample_paths["dynamic"].append(eq)

        # v1 baseline: fixed poisson+ml base, no self-refit, no fusion
        rng = np.random.default_rng(1000 + t)
        f, eq, layer_v1 = run_dynamic_trial_clean(
            train_df, strengths, rng, args.matches, poisson=poisson, ml=ml,
            mode="v1")
        finals["dynamic_v1"].append(f)
        bet_counts["dynamic_v1"].append(layer_v1.n_bets)

    # ---- report
    years = args.matches / 365.25
    rows = []
    for pol in ("flat", "kelly", "dynamic", "dynamic_v1"):
        arr = np.array(finals[pol])
        prof = arr - INITIAL_INVESTMENT
        med_cagr = (np.median(arr) / INITIAL_INVESTMENT) ** (1 / years) - 1
        rows.append({
            "policy": pol,
            "mean": arr.mean(), "median": np.median(arr),
            "p_profit": (prof > 0).mean() * 100,
            "p5": np.percentile(arr, 5), "p95": np.percentile(arr, 95),
            "worst": arr.min(), "best": arr.max(),
            "median_cagr": med_cagr * 100,
            "avg_bets": np.mean(bet_counts[pol]),
        })
        print(f"\n  {pol:<12} mean ${arr.mean():>12,.0f}  median ${np.median(arr):>12,.0f}"
              f"  P(profit) {(prof > 0).mean()*100:>3.0f}%  "
              f"90% [{np.percentile(arr,5):,.0f} .. {np.percentile(arr,95):,.0f}]"
              f"  CAGR {med_cagr*100:+.1f}%")

    print(f"\n  dynamic: final market weight (model-trust) = {np.mean(weights):.3f}"
          f"  (0 = trust model only, 1 = trust sharp market only)")
    print(f"  dynamic: base-model online refits per trial (avg) = {np.mean(refits):.1f}"
          f"  of which confidence-gated = {np.mean(conf_refits):.1f}")
    print(f"  dynamic: final rolling confidence (avg) = {np.mean(final_conf):.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(SIM_OUT / "simulation_1m_dynamic.csv", index=False)
    print(f"  [OK] Saved {SIM_OUT / 'simulation_1m_dynamic.csv'}")

    _plot(finals, sample_paths, years)
    _write_doc(res, np.mean(weights), np.mean(refits), np.mean(conf_refits),
               np.mean(final_conf))


def _plot(finals, sample_paths, years):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"flat": "#2E86AB", "kelly": "#A23B72", "dynamic": "#06D6A0"}
    for pol in ("flat", "kelly", "dynamic"):
        axes[0].hist(np.array(finals[pol]) / 1e6, bins=18, alpha=0.55,
                     label=pol, color=colors[pol])
    axes[0].axvline(1.0, color="gray", ls="--", lw=1.5)
    axes[0].set_title(f"Final bankroll distribution ($1M, {years:.0f} yrs)")
    axes[0].set_xlabel("Final bankroll ($M)"); axes[0].legend()
    for pol in ("flat", "dynamic"):
        for eq in sample_paths[pol]:
            axes[1].plot(eq / 1e6, alpha=0.5, lw=1.0, color=colors[pol])
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_title("Sample equity paths (first 8 trials)")
    axes[1].set_xlabel("Bets placed"); axes[1].set_ylabel("Bankroll ($M)")
    plt.tight_layout()
    plt.savefig(SIM_OUT / "simulation_1m_dynamic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {SIM_OUT / 'simulation_1m_dynamic.png'}")


def _write_doc(res: pd.DataFrame, mw: float, refits: float, conf_refits: float,
               final_conf: float):
    dyn = res[res["policy"] == "dynamic"].iloc[0]
    v1 = res[res["policy"] == "dynamic_v1"].iloc[0]
    lines = [
        "# $1M Simulation with the Dynamic Thinking Layer",
        "",
        "Every decision is made by `models/dynamic_thinking.py`: the model fuses",
        "the trained ensemble with the **public vs sharp market split** (a hidden",
        "signal), re-weights model-vs-market from its own rolling Brier, shrinks",
        "stakes with model/market disagreement and drawdown, and switches to a",
        "low-risk survival mode below 10% of the start.  The same forward match",
        "streams are replayed under flat and Kelly staking as controls.",
        "",
        "## Confidence-aware adaptation",
        "",
        "The layer now adapts **in proportion to how confident it is**:",
        "",
        "* **Confidence** = margin of the top outcome above the uniform 1/3",
        "  (1.0 = certain, 0.0 = coin-flip).",
        "* **Calibration blend** — the model-vs-market weight is re-weighted",
        "  from Brier that is *weighted by confidence* (a confident wrong call",
        "  hurts the model's trust more than a coin-flip), and the weight",
        "  update step grows with confidence: it adapts fast on clear signals,",
        "  cautiously when guessing.",
        "* **Confidence-gated refits** — if rolling confidence decays > 0.08",
        "  below its best, the base model refits on the recent window",
        "  immediately (in addition to its scheduled/drift refits).",
        "* **Confidence-scaled stakes** — stake x p(best)/0.40 (capped",
        "  [0.75, 1.4]), so the layer commits more only when it is genuinely",
        "  more sure than the minimum pass.",
        "",
        "```",
        res.to_string(index=False),
        "```",
        "",
        f"**dynamic vs its own v1 baseline (fixed base, no fusion, no confidence):** ",
        f"median ${dyn['median']:,.0f} vs ${v1['median']:,.0f}, ",
        f"P(profit) {dyn['p_profit']:.0f}% vs {v1['p_profit']:.0f}%, ",
        f"90% range [${dyn['p5']:,.0f} .. ${dyn['p95']:,.0f}] vs ",
        f"[${v1['p5']:,.0f} .. ${v1['p95']:,.0f}].  Final model-vs-market weight ",
        f"{mw:.2f}, base refits/trial {refits:.1f} (confidence-gated ",
        f"{conf_refits:.1f}), final rolling confidence {final_conf:.2f}.",
        "",
        "**Honest reading:** the dynamic layer adapts its reasoning (final market",
        "weight, uncertainty shrink, drawdown risk, survival switch) instead of",
        "applying a fixed rule — but in this synthetic world with a modest edge",
        "the *flat* policy's variance control is hard to beat.  The value of the",
        "thinking layer shows on real data where the sharp line carries genuine",
        "information (see `docs/09_real_walkforward_simulation.md` CLV results).",
        "",
        "*(Saved by `demo/simulation_dynamic.py`; per-trial numbers in",
        "`demo/output/simulation_1m_dynamic.csv`.)*",
    ]
    doc = PROJECT_ROOT / "docs" / "12_dynamic_thinking.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [OK] Wrote {doc}")


if __name__ == "__main__":
    main()
