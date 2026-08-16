#!/usr/bin/env python3
"""
Per-match analysis card - the "betting user" view of the system.

For every candidate match this module produces the full reasoning chain:

    model probability -> uncertainty -> bookmaker probability ->
    best odds -> fair odds -> edge -> EV -> bet/no-bet -> CLV -> result

It works on BOTH the synthetic schema (odds_*_b365 / odds_*_pin /
best_odds_*) and the real-data schema (odds_* / pin_* / best_odds_* from
football-data.co.uk Max columns), because the loader normalises both to the
same names.

Honesty rules built in:
    * bookmaker probabilities are the margin-removed implied probabilities
      (1/odds normalised), so the comparison is apples-to-apples;
    * edge is always reported against the BEST available price, never a
      cherry-picked book;
    * the uncertainty-adjusted rule only trusts an edge that exceeds z
      standard deviations of its own estimation error (winner's curse guard);
    * CLV is computed against the best closing line where available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402  (ensemble_probs, constants)
from models.calibration import implied_probs  # noqa: E402
from models.ml_layer import MLFootballPredictor  # noqa: E402
from models.poisson_elo_model import PoissonEloModel  # noqa: E402

OUTCOMES = ["home_win", "draw", "away_win"]
RESULT_TO_OUTCOME = {"H": "home_win", "D": "draw", "A": "away_win"}

# Column-name families, probed in order (synthetic first, real second).
_OPEN_BOOKS = {
    "b365": {"home_win": ["odds_home_b365", "odds_home"],
             "draw": ["odds_draw_b365", "odds_draw"],
             "away_win": ["odds_away_b365", "odds_away"]},
    "pin": {"home_win": ["odds_home_pin", "pin_home"],
            "draw": ["odds_draw_pin", "pin_draw"],
            "away_win": ["odds_away_pin", "pin_away"]},
}
_CLOSING_FAMILIES = {
    "home_win": ["best_closing_odds_home", "closing_odds_home"],
    "draw": ["best_closing_odds_draw", "closing_odds_draw"],
    "away_win": ["best_closing_odds_away", "closing_odds_away"],
}


def _probe(row: pd.Series, candidates: List[str],
           default: Optional[float] = None) -> Optional[float]:
    for c in candidates:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and float(v) > 1.0:
                return float(v)
    return default


def book_odds(row: pd.Series, book: str = "b365") -> Dict[str, Optional[float]]:
    fam = _OPEN_BOOKS.get(book, _OPEN_BOOKS["b365"])
    return {o: _probe(row, fam[o]) for o in OUTCOMES}


def available_books(row: pd.Series) -> List[str]:
    return [b for b in _OPEN_BOOKS if all(v is not None
                                          for v in book_odds(row, b).values())]


def best_odds(row: pd.Series) -> Dict[str, Optional[float]]:
    """Best available price per outcome across all books (price shopping)."""
    books = available_books(row)
    if not books:
        return {o: None for o in OUTCOMES}
    out = {}
    for o in OUTCOMES:
        vals = [book_odds(row, b)[o] for b in books]
        out[o] = float(np.max(vals))
    return out


def best_closing_odds(row: pd.Series) -> Dict[str, Optional[float]]:
    return {o: _probe(row, _CLOSING_FAMILIES[o]) for o in OUTCOMES}


def bookie_implied(row: pd.Series) -> Dict[str, float]:
    """Margin-removed bookmaker probabilities from the best prices."""
    best = best_odds(row)
    ip = implied_probs(best)
    return {o: ip.get(o, 1 / 3) for o in OUTCOMES}


def _quarter_kelly(edge: float, odds: float) -> float:
    if edge <= 0 or odds <= 1:
        return 0.0
    return min((edge / (odds - 1)) * 0.25, 0.05)


def analyze_match(
    row: pd.Series,
    poisson: PoissonEloModel,
    ml: Optional[MLFootballPredictor],
    edge_threshold: float = pipeline.EDGE_THRESHOLD,
    min_odds: float = pipeline.MIN_ODDS,
    min_model_prob: float = pipeline.MIN_MODEL_PROB,
    uncertainty_z: float = 0.0,
    n_samples: int = 200,
    seed: int = 0,
) -> dict:
    """Full reasoning card for one match.

    ``uncertainty_z > 0`` enables the uncertainty-adjusted decision: a bet is
    only placed when the edge exceeds the threshold AND remains positive after
    subtracting ``z`` standard deviations of the edge estimate (computed from
    the Poisson parameter-uncertainty Monte-Carlo).
    """
    home, away = row["home_team"], row["away_team"]
    p_model = pipeline.ensemble_probs(poisson, ml, home, away)
    u = poisson.predict_with_uncertainty(home, away, n_samples=n_samples, seed=seed)
    bookie = bookie_implied(row)
    best = best_odds(row)
    closing = best_closing_odds(row)

    edges = {}
    for o in OUTCOMES:
        edges[o] = (p_model[o] * best[o] - 1.0) if best[o] else None
    best_outcome = max(OUTCOMES, key=lambda o: edges[o] if edges[o] is not None else -1)
    edge = edges[best_outcome]
    odds = best[best_outcome]

    # uncertainty of the edge estimate: d(edge)/d(p) = odds
    sigma_edge = odds * u[f"{best_outcome}_std"] if odds else 0.0

    # ---- decision ---------------------------------------------------------
    reasons = []
    decision = "BET"
    if edge is None or odds is None:
        decision, reasons = "NO_BET", ["no odds available"]
    else:
        if edge <= edge_threshold:
            decision, reasons = "NO_BET", [f"edge {edge:.1%} <= threshold {edge_threshold:.1%}"]
        if odds < min_odds:
            decision, reasons = "NO_BET", [f"odds {odds:.2f} < min {min_odds:.2f}"]
        if p_model[best_outcome] < min_model_prob:
            decision, reasons = "NO_BET", [
                f"model prob {p_model[best_outcome]:.1%} < min {min_model_prob:.1%}"]
        if uncertainty_z > 0 and edge > edge_threshold:
            if edge - uncertainty_z * sigma_edge <= 0:
                decision, reasons = "NO_BET", [
                    f"edge {edge:.1%} < {uncertainty_z:.1f} sigma ({sigma_edge:.1%}) - "
                    "uncertainty-adjusted"]

    clv = ((closing[best_outcome] - odds) / odds) if (
        closing[best_outcome] and odds) else None
    fair_model = 1.0 / p_model[best_outcome] if p_model[best_outcome] > 0 else None
    fair_bookie = 1.0 / bookie[best_outcome] if bookie[best_outcome] > 0 else None

    return {
        "date": str(row.get("date", "")),
        "home_team": home, "away_team": away,
        "league": row.get("league", ""), "season": row.get("season", ""),
        # model probabilities and uncertainty
        "p_model_home_win": round(p_model["home_win"], 4),
        "p_model_draw": round(p_model["draw"], 4),
        "p_model_away_win": round(p_model["away_win"], 4),
        "unc_home_win": round(u["home_win_std"], 4),
        "unc_draw": round(u["draw_std"], 4),
        "unc_away_win": round(u["away_win_std"], 4),
        # bookmaker probabilities (margin removed)
        "p_bookie_home_win": round(bookie["home_win"], 4),
        "p_bookie_draw": round(bookie["draw"], 4),
        "p_bookie_away_win": round(bookie["away_win"], 4),
        # prices and fair odds
        "best_odds_home_win": best["home_win"],
        "best_odds_draw": best["draw"],
        "best_odds_away_win": best["away_win"],
        "fair_odds_model": round(fair_model, 3) if fair_model else None,
        "fair_odds_bookie": round(fair_bookie, 3) if fair_bookie else None,
        # decision chain
        "best_outcome": best_outcome,
        "edge_pct": round(edge * 100, 2) if edge is not None else None,
        "edge_uncertainty_pct": round(sigma_edge * 100, 2),
        "ev_per_unit_pct": round(edge * 100, 2) if edge is not None else None,
        "decision": decision,
        "reason": "; ".join(reasons),
        "kelly_stake_frac": round(_quarter_kelly(edge, odds), 4) if (
            decision == "BET" and edge and odds) else 0.0,
        "clv_pct": round(clv * 100, 2) if clv is not None else None,
        "result": row.get("result", ""),
        "correct": bool(RESULT_TO_OUTCOME.get(row.get("result", "")) == best_outcome),
        "n_books": len(available_books(row)),
    }


def build_predictions_table(df: pd.DataFrame, poisson: PoissonEloModel,
                            ml: Optional[MLFootballPredictor],
                            uncertainty_z: float = 0.0,
                            n_samples: int = 200,
                            seed: int = 0) -> pd.DataFrame:
    """Score a whole dataframe into the transparent per-match table."""
    rows = []
    for _, row in df.iterrows():
        rows.append(analyze_match(row, poisson, ml, uncertainty_z=uncertainty_z,
                                  n_samples=n_samples, seed=seed))
    return pd.DataFrame(rows)


def write_predictions_table(df: pd.DataFrame, poisson: PoissonEloModel,
                            ml: Optional[MLFootballPredictor],
                            path, **kwargs) -> pd.DataFrame:
    """Build and save the predictions table; returns it for convenience."""
    table = build_predictions_table(df, poisson, ml, **kwargs)
    path = str(path)
    table.to_csv(path, index=False)
    print(f"  [OK] Saved predictions table -> {path} ({len(table)} matches)")
    return table


if __name__ == "__main__":
    # Self-test on the synthetic world.
    data, _ = pipeline.load_or_generate_data(n_matches=600, seed=42)
    poisson = pipeline.PoissonEloModel(use_dixon_coles=False)
    poisson.train(data.iloc[:420], verbose=False)
    card = analyze_match(data.iloc[500], poisson, None)
    for k, v in card.items():
        print(f"  {k:<22}: {v}")
    assert sum(card[k] for k in ("p_model_home_win", "p_model_draw",
                                 "p_model_away_win")) > 0.99
    print("[OK] match_analysis self-test passed.")
