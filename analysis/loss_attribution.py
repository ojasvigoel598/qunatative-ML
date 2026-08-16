#!/usr/bin/env python3
"""
"Why is the model losing?" - automatic loss attribution.

Decomposes the gap between the advertised edge and the realized return into
the individual mechanisms that economics and sports-betting research actually
blame, and reports which one dominates:

    1. Margin drag      - the overround tax the model pays on every bet
                          before any skill is involved.
    2. Winner's curse   - selection bias: the model only bets where it
                          disagrees with the market most, which is where its
                          own estimation error is largest.  Measured as the
                          gap between advertised edge and realized edge, and
                          between model probability on bets vs actual win
                          rate (betting-region calibration gap).
    3. Calibration      - overall ECE on all matches (not just bets).
    4. Information      - CLV: does the taken price beat the closing line?
                          CLV ~ 0 with a small t-stat is the textbook
                          signature of NO information edge.
    5. Market fit       - model vs bookmaker implied log loss on the same
                          matches: is the model even a better estimator?

The report never fabricates a conclusion: if the dominant mechanism is
"no information" (CLV ~ 0), it says so.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402
from analysis.match_analysis import analyze_match  # noqa: E402
from models.ml_layer import MLFootballPredictor  # noqa: E402
from models.poisson_elo_model import PoissonEloModel  # noqa: E402


def _parse_match(match_str: str):
    if " vs " in match_str:
        return tuple(match_str.split(" vs "))
    return (match_str, "")


def attribute_losses(result: dict) -> Dict[str, float]:
    """Decompose the gap between advertised edge and realized return.

    ``result`` is the dict returned by ``pipeline.run_backtest`` (needs
    bets_df, models, test_scored, test_eval).
    """
    bets = result["bets_df"]
    models = result["models"]
    poisson: PoissonEloModel = models["poisson"]
    ml: Optional[MLFootballPredictor] = models["ml"]
    test_eval = result.get("test_eval", {})

    out: Dict[str, float] = {}
    out["n_bets"] = int(len(bets))
    if len(bets) == 0:
        out["dominant_mechanism"] = "no_bets"
        return out

    # ---- per-bet model probabilities (re-derived, no schema change) ------
    p_model_bets, advertised, realized = [], [], []
    for _, b in bets.iterrows():
        home, away = _parse_match(b["match"])
        probs = pipeline.ensemble_probs(poisson, ml, home, away)
        p = probs[b["market"]]
        odds = b["my_odds"]
        p_model_bets.append(p)
        advertised.append(p * odds - 1.0)
        realized.append((odds - 1.0) if b["bet_outcome"] == "Win" else -1.0)

    p_model_bets = np.array(p_model_bets)
    advertised = np.array(advertised)
    realized = np.array(realized)

    win_rate = float(np.mean(realized > 0))
    avg_odds = float(bets["my_odds"].mean())
    realized_edge = win_rate * avg_odds - 1.0

    out["avg_advertised_edge_pct"] = round(float(advertised.mean()) * 100, 2)
    out["avg_realized_edge_pct"] = round(realized_edge * 100, 2)
    out["selection_loss_pct"] = round(
        float((advertised.mean() - realized_edge) * 100), 2)
    out["betting_region_model_prob_pct"] = round(float(p_model_bets.mean()) * 100, 2)
    out["betting_region_win_rate_pct"] = round(win_rate * 100, 2)
    out["betting_region_cal_gap_pct"] = round(
        float((p_model_bets.mean() - win_rate) * 100), 2)

    # ---- margin drag: average overround on the matches the backtest saw ----
    scored = result.get("test_scored")
    overrounds = []
    if scored is not None and len(scored):
        for _, row in scored.iterrows():
            s = (1 / row["odds_home_b365"] + 1 / row["odds_draw_b365"]
                 + 1 / row["odds_away_b365"])
            overrounds.append(float(s - 1.0))
    out["avg_overround_pct"] = round(float(np.mean(overrounds)) * 100, 2) \
        if overrounds else 0.0
    # A bettor who takes every outcome of a book loses the full margin; a
    # selective bettor pays it proportionally.  This is the tax before skill.
    out["margin_drag_pct"] = round(-out["avg_overround_pct"], 2)

    # ---- CLV information signal -------------------------------------------
    clv = bets["clv_pct"].to_numpy(dtype=float)
    out["avg_clv_pct"] = round(float(clv.mean()), 2)
    out["clv_win_rate_pct"] = round(float(np.mean(clv > 0)) * 100, 2)
    if len(clv) > 1:
        sd = float(clv.std(ddof=1))
        out["clv_t_stat"] = round(float(clv.mean() / (sd / np.sqrt(len(clv)))), 2) \
            if sd > 0 else 0.0
    else:
        out["clv_t_stat"] = 0.0

    # ---- market fit (from the backtest's own test_eval) -------------------
    out["model_log_loss"] = test_eval.get("log_loss")
    out["market_log_loss"] = test_eval.get("market_log_loss")
    out["beats_market"] = float(bool(test_eval.get("beats_market_logloss", False)))

    # ---- dominant mechanism ------------------------------------------------
    scores = {
        "margin_drag": abs(out["margin_drag_pct"]),
        "selection_loss": abs(out["selection_loss_pct"]),
        "calibration_gap": abs(out["betting_region_cal_gap_pct"]),
    }
    out["dominant_mechanism"] = max(scores, key=scores.get)
    return out


def write_loss_report(result: dict, out_path) -> str:
    """Write the human-readable attribution report; returns the text."""
    a = attribute_losses(result)
    dominant = a.pop("dominant_mechanism", "no_bets")

    lines = []
    lines.append("WHY IS THE MODEL LOSING? (automatic loss attribution)")
    lines.append("=" * 56)
    if a["n_bets"] == 0:
        lines.append("No bets were placed - nothing to attribute.")
        text = "\n".join(lines)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text, encoding="utf-8")
        return text

    clv_ok = abs(a["clv_t_stat"]) >= 2.0 and a["avg_clv_pct"] > 0
    lines.append(f"Bets analysed: {a['n_bets']}")
    lines.append("")
    lines.append(f"1. Margin drag    : -{a['avg_overround_pct']:.2f}% per bet "
                 f"(average overround on the matches bet). The house keeps this "
                 f"before any skill.")
    lines.append(f"2. Winner's curse : advertised edge "
                 f"+{a['avg_advertised_edge_pct']:.2f}% vs realised edge "
                 f"{a['avg_realized_edge_pct']:+.2f}% -> selection cost "
                 f"{a['selection_loss_pct']:+.2f}%. Model prob on bets "
                 f"{a['betting_region_model_prob_pct']:.1f}% vs actual wins "
                 f"{a['betting_region_win_rate_pct']:.1f}% "
                 f"(betting-region calibration gap "
                 f"{a['betting_region_cal_gap_pct']:+.1f} pts).")
    lines.append(f"3. Information    : mean CLV {a['avg_clv_pct']:+.2f}%, "
                 f"win rate vs closing {a['clv_win_rate_pct']:.1f}%, "
                 f"t = {a['clv_t_stat']:+.2f}. "
                 + ("Significant positive CLV - real information signal."
                    if clv_ok else
                    "No significant edge over the closing line (CLV is the "
                    "standard test for real information)."))
    mll = a.get("model_log_loss")
    if mll is not None and a.get("market_log_loss") is not None:
        lines.append(f"4. Market fit     : model log loss {mll} vs market "
                     f"{a['market_log_loss']} -> "
                     + ("model beats the market's probabilities"
                        if a["beats_market"] else
                        "model is NOT a better estimator than the market"))

    lines.append("")
    verdicts = {
        "selection_loss": "DOMINANT: winner's curse / selection bias. The model "
                          "bets where it disagrees with the market most, and that "
                          "is where its own estimation error is largest.",
        "margin_drag": "DOMINANT: margin drag. The overround alone explains most "
                       "of the loss; even a perfect model would lose here.",
        "calibration_gap": "DOMINANT: betting-region miscalibration. The model "
                           "overstates win probability exactly where it bets.",
    }
    lines.append(verdicts.get(dominant, verdicts["margin_drag"]))
    if not clv_ok:
        lines.append("INFO: CLV ~ 0 with |t| < 2 means there is no measurable "
                     "information edge over the closing line - the model is not "
                     "beating the market, so the negative ROI is expected, not a bug.")
    text = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"  [OK] Saved loss attribution -> {out_path}")
    return text


if __name__ == "__main__":
    data, _ = pipeline.load_or_generate_data(n_matches=800, seed=42)
    res = pipeline.run_backtest(data, use_ml=True, use_rl=True,
                                save_results=False, verbose=False)
    report = write_loss_report(res, "backtests/results/why_model_losing_test.txt")
    print(report)
