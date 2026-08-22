#!/usr/bin/env python3
"""
Real-Time Odds Tracking

Compare opening vs closing odds to measure:
- Line movement (did odds shift?)
- Sharp money detection (did sharp bettors move the line?)
- Market efficiency (how quickly does info get priced in?)

Key insight: If your model predicts differently from the closing line,
you have genuine edge. If you match the closing line, the market already
knows what you know.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class OddsSnapshot:
    """A single odds observation at a point in time."""
    timestamp: str
    bookmaker: str
    home_odds: float
    draw_odds: float
    away_odds: float
    source: str = "opening"  # "opening", "live", "closing"


@dataclass
class OddsMovement:
    """Track odds movement for a single match."""
    match_id: str
    home_team: str
    away_team: str
    date: str
    
    # Opening odds
    opening_home: float
    opening_draw: float
    opening_away: float
    
    # Closing odds
    closing_home: float
    closing_draw: float
    closing_away: float
    
    # Movement (positive = odds drifted out, negative = odds shortened)
    movement_home: float = 0.0
    movement_draw: float = 0.0
    movement_away: float = 0.0
    
    # Market margin
    opening_margin: float = 0.0
    closing_margin: float = 0.0
    
    # Sharp money indicator
    sharp_money_home: bool = False
    sharp_money_draw: bool = False
    sharp_money_away: bool = False


@dataclass
class OddsTrackerSummary:
    """Summary of odds tracking analysis."""
    n_matches: int = 0
    avg_movement: float = 0.0
    avg_closing_margin: float = 0.0
    sharp_money_pct: float = 0.0
    line_move_direction: str = ""  # "home", "away", "neutral"
    market_efficiency: float = 0.0


def compute_odds_movement(
    opening_home: float,
    opening_draw: float,
    opening_away: float,
    closing_home: float,
    closing_draw: float,
    closing_away: float
) -> Dict[str, float]:
    """Compute odds movement between opening and closing.
    
    Returns:
        Dictionary with movement for each outcome
    """
    movement_home = closing_home - opening_home
    movement_draw = closing_draw - opening_draw
    movement_away = closing_away - opening_away
    
    return {
        "home": movement_home,
        "draw": movement_draw,
        "away": movement_away,
    }


def compute_market_margin(home_odds: float, draw_odds: float, away_odds: float) -> float:
    """Compute bookmaker margin (overround).
    
    Margin = (1/home + 1/draw + 1/away) - 1
    
    Lower margin = more efficient market.
    Typical margins: 2-10%
    """
    return (1/home_odds + 1/draw_odds + 1/away_odds) - 1


def detect_sharp_money(
    opening_odds: float,
    closing_odds: float,
    threshold: float = 0.1
) -> bool:
    """Detect if sharp money moved the line.
    
    Sharp money typically:
    - Moves odds significantly (>10%)
    - Moves in one direction consistently
    - Happens close to kick-off
    
    Args:
        opening_odds: Opening odds
        closing_odds: Closing odds
        threshold: Minimum movement to consider "sharp"
    
    Returns:
        True if sharp money detected
    """
    if opening_odds <= 0:
        return False
    
    pct_change = abs(closing_odds - opening_odds) / opening_odds
    return pct_change > threshold


def analyze_odds_movement(
    matches: List[Dict],
    verbose: bool = True
) -> OddsTrackerSummary:
    """Analyze odds movement across multiple matches.
    
    Args:
        matches: List of match dicts with opening/closing odds
        verbose: Print analysis
    
    Returns:
        OddsTrackerSummary
    """
    movements = []
    margins = []
    sharp_counts = {"home": 0, "draw": 0, "away": 0}
    total = 0
    
    for match in matches:
        opening = match.get("opening_odds", {})
        closing = match.get("closing_odds", {})
        
        if not opening or not closing:
            continue
        
        move = compute_odds_movement(
            opening.get("home", 0), opening.get("draw", 0), opening.get("away", 0),
            closing.get("home", 0), closing.get("draw", 0), closing.get("away", 0)
        )
        
        movements.append(move)
        
        # Closing margin
        margin = compute_market_margin(
            closing.get("home", 2), closing.get("draw", 3), closing.get("away", 3)
        )
        margins.append(margin)
        
        # Sharp money detection
        for outcome in ["home", "draw", "away"]:
            if detect_sharp_money(
                opening.get(outcome, 2),
                closing.get(outcome, 2)
            ):
                sharp_counts[outcome] += 1
        
        total += 1
    
    if total == 0:
        return OddsTrackerSummary()
    
    avg_movement = np.mean([abs(m["home"]) + abs(m["draw"]) + abs(m["away"]) 
                           for m in movements])
    avg_margin = np.mean(margins)
    total_sharp = sum(sharp_counts.values())
    sharp_pct = total_sharp / (total * 3)
    
    # Determine line move direction
    home_moves = [m["home"] for m in movements]
    away_moves = [m["away"] for m in movements]
    
    if np.mean(home_moves) < 0:
        direction = "home (odds shortened = money on home)"
    elif np.mean(away_moves) < 0:
        direction = "away (odds shortened = money on away)"
    else:
        direction = "neutral"
    
    summary = OddsTrackerSummary(
        n_matches=total,
        avg_movement=round(avg_movement, 4),
        avg_closing_margin=round(avg_margin, 4),
        sharp_money_pct=round(sharp_pct, 4),
        line_move_direction=direction,
        market_efficiency=round(1 - avg_margin, 4),
    )
    
    if verbose:
        print(f"Odds Movement Analysis ({total} matches):")
        print(f"  Average movement: {summary.avg_movement:.4f}")
        print(f"  Closing margin: {summary.avg_closing_margin:.1%}")
        print(f"  Sharp money: {summary.sharp_money_pct:.1%}")
        print(f"  Line direction: {summary.line_move_direction}")
        print(f"  Market efficiency: {summary.market_efficiency:.1%}")
    
    return summary


def compare_opening_vs_closing(
    opening_odds: Dict[str, float],
    closing_odds: Dict[str, float],
    model_probs: Dict[str, float]
) -> Dict[str, float]:
    """Compare model predictions against opening and closing odds.
    
    This tells you:
    1. Did your model agree with opening odds? (early edge)
    2. Did your model agree with closing odds? (late edge)
    3. Did the market move toward your model? (info incorporation)
    
    Returns:
        Dictionary with comparison metrics
    """
    # Convert odds to probabilities
    opening_probs = {
        "home": 1/opening_odds["home"],
        "draw": 1/opening_odds["draw"],
        "away": 1/opening_odds["away"],
    }
    closing_probs = {
        "home": 1/closing_odds["home"],
        "draw": 1/closing_odds["draw"],
        "away": 1/closing_odds["away"],
    }
    
    # Compute differences
    diff_from_opening = {
        "home": model_probs["home"] - opening_probs["home"],
        "draw": model_probs["draw"] - opening_probs["draw"],
        "away": model_probs["away"] - opening_probs["away"],
    }
    
    diff_from_closing = {
        "home": model_probs["home"] - closing_probs["home"],
        "draw": model_probs["draw"] - closing_probs["draw"],
        "away": model_probs["away"] - closing_probs["away"],
    }
    
    # Did market move toward model?
    market_moved_toward_model = {
        "home": diff_from_opening["home"] > diff_from_closing["home"],
        "draw": diff_from_opening["draw"] > diff_from_closing["draw"],
        "away": diff_from_opening["away"] > diff_from_closing["away"],
    }
    
    return {
        "model_probs": model_probs,
        "opening_probs": opening_probs,
        "closing_probs": closing_probs,
        "diff_from_opening": diff_from_opening,
        "diff_from_closing": diff_from_closing,
        "market_moved_toward_model": market_moved_toward_model,
    }


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    print("Testing odds tracker module...")
    
    # Test margin computation
    margin = compute_market_margin(2.0, 3.5, 3.8)
    print(f"  Market margin: {margin:.3f} ({margin:.1%})")
    
    # Test sharp money detection
    sharp = detect_sharp_money(2.0, 1.7, threshold=0.1)
    print(f"  Sharp money detected: {sharp}")
    
    # Test comparison
    comparison = compare_opening_vs_closing(
        opening_odds={"home": 2.0, "draw": 3.5, "away": 3.8},
        closing_odds={"home": 1.8, "draw": 3.6, "away": 4.2},
        model_probs={"home": 0.55, "draw": 0.25, "away": 0.20}
    )
    print(f"  Model vs opening (home): {comparison['diff_from_opening']['home']:.3f}")
    print(f"  Model vs closing (home): {comparison['diff_from_closing']['home']:.3f}")
    print(f"  Market moved toward model: {comparison['market_moved_toward_model']['home']}")
    
    print("\n[OK] Odds tracker module complete.")
