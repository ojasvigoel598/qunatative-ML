#!/usr/bin/env python3
"""
Closing Line Value (CLV) Tracking

CLV measures whether your model's predictions improve over time.
If your model consistently beats the closing line, it has genuine edge.

CLV = (your_model_probability - closing_market_probability) / closing_market_probability

Positive CLV = your model is better than the closing market
Negative CLV = your model is worse than the closing market

Key insight from research: CLV is the best predictor of long-term profitability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CLVResult:
    """CLV tracking result for a single bet."""
    match_id: str
    home_team: str
    away_team: str
    date: str
    outcome: str  # "H", "D", "A"
    
    # Model predictions
    model_home_prob: float
    model_draw_prob: float
    model_away_prob: float
    
    # Opening odds (when you placed bet)
    opening_home_odds: float
    opening_draw_odds: float
    opening_away_odds: float
    
    # Closing odds (at kick-off)
    closing_home_odds: float
    closing_draw_odds: float
    closing_away_odds: float
    
    # CLV for each outcome
    clv_home: float = 0.0
    clv_draw: float = 0.0
    clv_away: float = 0.0
    
    # Average CLV across outcomes you bet on
    avg_clv: float = 0.0
    
    # Did the line move in your favor?
    line_moved_for_you: bool = False


@dataclass 
class CLVSummary:
    """Summary of CLV tracking across many bets."""
    n_bets: int = 0
    avg_clv: float = 0.0
    positive_clv_pct: float = 0.0  # % of bets with positive CLV
    clv_by_outcome: Dict[str, float] = field(default_factory=dict)
    clv_trend: List[float] = field(default_factory=list)  # CLV over time
    correlation_with_profit: float = 0.0
    interpretation: str = ""


def compute_clv(
    model_prob: float,
    opening_odds: float,
    closing_odds: float
) -> float:
    """Compute Closing Line Value for a single bet.
    
    CLV = (model_prob - implied_prob_closing) / implied_prob_closing
    
    Args:
        model_prob: Your model's predicted probability
        opening_odds: Odds when you placed the bet
        closing_odds: Odds at kick-off
    
    Returns:
        CLV value (positive = line moved in your favor)
    """
    if closing_odds <= 1.0 or model_prob <= 0:
        return 0.0
    
    implied_prob_closing = 1.0 / closing_odds
    
    if implied_prob_closing <= 0:
        return 0.0
    
    clv = (model_prob - implied_prob_closing) / implied_prob_closing
    return clv


def compute_clv_for_match(
    model_probs: Dict[str, float],
    opening_odds: Dict[str, float],
    closing_odds: Dict[str, float],
    bet_outcome: str = "home"
) -> Dict[str, float]:
    """Compute CLV for all outcomes of a match.
    
    Args:
        model_probs: {"home": 0.5, "draw": 0.25, "away": 0.25}
        opening_odds: {"home": 2.0, "draw": 3.5, "away": 3.8}
        closing_odds: {"home": 1.9, "draw": 3.6, "away": 4.0}
        bet_outcome: Which outcome was bet on
    
    Returns:
        Dictionary with CLV for each outcome
    """
    clv_home = compute_clv(model_probs.get("home", 0), 
                           opening_odds.get("home", 0),
                           closing_odds.get("home", 0))
    clv_draw = compute_clv(model_probs.get("draw", 0),
                           opening_odds.get("draw", 0), 
                           closing_odds.get("draw", 0))
    clv_away = compute_clv(model_probs.get("away", 0),
                           opening_odds.get("away", 0),
                           closing_odds.get("away", 0))
    
    # Determine which CLV to average based on bet
    if bet_outcome == "home":
        avg_clv = clv_home
    elif bet_outcome == "draw":
        avg_clv = clv_draw
    else:
        avg_clv = clv_away
    
    return {
        "clv_home": clv_home,
        "clv_draw": clv_draw,
        "clv_away": clv_away,
        "avg_clv": avg_clv,
    }


def track_clv_batch(
    predictions: List[Dict],
    verbose: bool = True
) -> CLVSummary:
    """Track CLV for a batch of predictions.
    
    Args:
        predictions: List of dicts with model_probs, opening_odds, closing_odds
        verbose: Print summary
    
    Returns:
        CLVSummary
    """
    results = []
    
    for pred in predictions:
        model_probs = pred["model_probs"]
        opening_odds = pred["opening_odds"]
        closing_odds = pred["closing_odds"]
        bet_outcome = pred.get("bet_outcome", "home")
        
        clv_dict = compute_clv_for_match(
            model_probs, opening_odds, closing_odds, bet_outcome
        )
        
        results.append(clv_dict)
    
    if not results:
        return CLVSummary()
    
    avg_clv = np.mean([r["avg_clv"] for r in results])
    positive_pct = sum(1 for r in results if r["avg_clv"] > 0) / len(results)
    
    # CLV by outcome
    clv_by_outcome = {
        "home": np.mean([r["clv_home"] for r in results]),
        "draw": np.mean([r["clv_draw"] for r in results]),
        "away": np.mean([r["clv_away"] for r in results]),
    }
    
    # Interpretation
    if avg_clv > 0.05:
        interpretation = "EXCELLENT: Strong edge over closing line"
    elif avg_clv > 0.02:
        interpretation = "GOOD: Consistent edge over closing line"
    elif avg_clv > 0:
        interpretation = "MARGINAL: Slight edge over closing line"
    else:
        interpretation = "POOR: Model is worse than closing line"
    
    summary = CLVSummary(
        n_bets=len(results),
        avg_clv=round(avg_clv, 4),
        positive_clv_pct=round(positive_pct, 4),
        clv_by_outcome=clv_by_outcome,
        interpretation=interpretation,
    )
    
    if verbose:
        print(f"CLV Summary ({summary.n_bets} bets):")
        print(f"  Average CLV: {summary.avg_clv:.4f}")
        print(f"  Positive CLV: {summary.positive_clv_pct:.1%}")
        print(f"  Home CLV: {clv_by_outcome['home']:.4f}")
        print(f"  Draw CLV: {clv_by_outcome['draw']:.4f}")
        print(f"  Away CLV: {clv_by_outcome['away']:.4f}")
        print(f"  {summary.interpretation}")
    
    return summary


def clv_report(
    results: List[CLVResult],
    verbose: bool = True
) -> str:
    """Generate detailed CLV report.
    
    Args:
        results: List of CLVResult objects
        verbose: Print report
    
    Returns:
        Formatted report string
    """
    if not results:
        return "No CLV data available."
    
    avg_clv = np.mean([r.avg_clv for r in results])
    positive_pct = sum(1 for r in results if r.avg_clv > 0) / len(results)
    
    report = f"""
{'='*60}
CLOSING LINE VALUE (CLV) REPORT
{'='*60}

Total Bets Tracked: {len(results)}
Average CLV:        {avg_clv:.4f}
Positive CLV:       {positive_pct:.1%}

Interpretation:
  CLV > 0.05:  EXCELLENT - Strong edge
  CLV > 0.02:  GOOD - Consistent edge
  CLV > 0.00:  MARGINAL - Slight edge
  CLV < 0.00:  POOR - No edge

Your Average CLV: {avg_clv:.4f}
{'='*60}
"""
    
    if verbose:
        print(report)
    
    return report


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    print("Testing CLV tracking module...")
    
    # Example: Model says 60% home win, market opened at 2.0, closed at 1.8
    model_prob = 0.6
    opening_odds = 2.0
    closing_odds = 1.8
    
    clv = compute_clv(model_prob, opening_odds, closing_odds)
    print(f"  Model prob: {model_prob:.2f}")
    print(f"  Opening odds: {opening_odds:.2f}")
    print(f"  Closing odds: {closing_odds:.2f}")
    print(f"  CLV: {clv:.4f}")
    print(f"  Interpretation: {'GOOD' if clv > 0 else 'BAD'}")
    
    # Batch tracking
    predictions = [
        {
            "model_probs": {"home": 0.6, "draw": 0.25, "away": 0.15},
            "opening_odds": {"home": 2.0, "draw": 3.5, "away": 5.0},
            "closing_odds": {"home": 1.8, "draw": 3.6, "away": 5.5},
            "bet_outcome": "home",
        },
        {
            "model_probs": {"home": 0.3, "draw": 0.3, "away": 0.4},
            "opening_odds": {"home": 3.5, "draw": 3.2, "away": 2.2},
            "closing_odds": {"home": 3.8, "draw": 3.0, "away": 2.1},
            "bet_outcome": "away",
        },
    ]
    
    summary = track_clv_batch(predictions)
    print(f"\n  Average CLV: {summary.avg_clv:.4f}")
    print(f"  Positive: {summary.positive_clv_pct:.1%}")
    
    print("\n[OK] CLV tracking module complete.")
