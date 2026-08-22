#!/usr/bin/env python3
"""
Alternative Markets — Totals, Asian Handicaps

Research shows alternative markets (totals, Asian handicaps) often have
higher bookmaker margins but also more opportunities for edge.

Key insight: Bookmakers are less efficient in alternative markets
because fewer bettors play them, so the odds are less sharp.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class MarketOdds:
    """Odds for a single market."""
    market_type: str  # "match_result", "totals", "asian_handicap"
    odds: Dict[str, float]  # outcome -> odds
    margin: float  # bookmaker margin
    implied_probs: Dict[str, float]  # outcome -> implied probability


@dataclass
class BetOpportunity:
    """A betting opportunity with edge."""
    market_type: str
    outcome: str
    model_prob: float
    market_prob: float
    odds: float
    edge: float  # model_prob - market_prob
    kelly_stake: float
    confidence: str  # "high", "medium", "low"


# ======================================================================
# Totals Market (Over/Under)
# ======================================================================
def totals_market(
    expected_total_goals: float,
    line: float = 2.5,
    over_odds: float = 1.9,
    under_odds: float = 1.9
) -> MarketOdds:
    """Analyze totals (over/under) market.
    
    Args:
        expected_total_goals: Model's expected total goals
        line: Total goals line (e.g., 2.5)
        over_odds: Odds for over
        under_odds: Odds for under
    
    Returns:
        MarketOdds with analysis
    """
    # Model probabilities using Poisson
    from scipy.stats import poisson
    
    p_under = poisson.cdf(int(line), expected_total_goals)
    p_over = 1 - p_under
    
    # Market implied probabilities
    margin = (1/over_odds + 1/under_odds) - 1
    implied_over = (1/over_odds) / (1 + margin)
    implied_under = (1/under_odds) / (1 + margin)
    
    return MarketOdds(
        market_type="totals",
        odds={"over": over_odds, "under": under_odds},
        margin=margin,
        implied_probs={"over": implied_over, "under": implied_under}
    )


def totals_edge(
    expected_total_goals: float,
    line: float = 2.5,
    over_odds: float = 1.9,
    under_odds: float = 1.9,
    min_edge: float = 0.05
) -> Optional[BetOpportunity]:
    """Find edge in totals market.
    
    Returns:
        BetOpportunity if edge exists, None otherwise
    """
    market = totals_market(expected_total_goals, line, over_odds, under_odds)
    
    model_over = market.implied_probs["over"]
    model_under = market.implied_probs["under"]
    
    # Check for edge
    edge_over = model_over - market.implied_probs["over"]
    edge_under = model_under - market.implied_probs["under"]
    
    if edge_over > min_edge:
        kelly = (model_over * over_odds - 1) / (over_odds - 1)
        return BetOpportunity(
            market_type="totals",
            outcome="over",
            model_prob=model_over,
            market_prob=market.implied_probs["over"],
            odds=over_odds,
            edge=edge_over,
            kelly_stake=max(kelly, 0),
            confidence="high" if edge_over > 0.1 else "medium"
        )
    
    if edge_under > min_edge:
        kelly = (model_under * under_odds - 1) / (under_odds - 1)
        return BetOpportunity(
            market_type="totals",
            outcome="under",
            model_prob=model_under,
            market_prob=market.implied_probs["under"],
            odds=under_odds,
            edge=edge_under,
            kelly_stake=max(kelly, 0),
            confidence="high" if edge_under > 0.1 else "medium"
        )
    
    return None


# ======================================================================
# Asian Handicap Market
# ======================================================================
def asian_handicap_market(
    expected_goal_diff: float,
    handicap: float = -1.0,
    home_odds: float = 1.9,
    away_odds: float = 1.9
) -> MarketOdds:
    """Analyze Asian Handicap market.
    
    Args:
        expected_goal_diff: Model's expected goal difference (home - away)
        handicap: Handicap (negative = home favored)
        home_odds: Odds for home + handicap
        away_odds: Odds for away - handicap
    
    Returns:
        MarketOdds with analysis
    """
    # Model probabilities
    adjusted_diff = expected_goal_diff + handicap  # apply handicap
    
    # Use normal approximation for goal difference
    from scipy.stats import norm
    p_home = norm.cdf(adjusted_diff, loc=0, scale=1.1)
    p_away = 1 - p_home
    
    # Market implied probabilities
    margin = (1/home_odds + 1/away_odds) - 1
    implied_home = (1/home_odds) / (1 + margin)
    implied_away = (1/away_odds) / (1 + margin)
    
    return MarketOdds(
        market_type="asian_handicap",
        odds={"home": home_odds, "away": away_odds},
        margin=margin,
        implied_probs={"home": implied_home, "away": implied_away}
    )


def asian_handicap_edge(
    expected_goal_diff: float,
    handicap: float = -1.0,
    home_odds: float = 1.9,
    away_odds: float = 1.9,
    min_edge: float = 0.05
) -> Optional[BetOpportunity]:
    """Find edge in Asian Handicap market.
    
    Returns:
        BetOpportunity if edge exists, None otherwise
    """
    market = asian_handicap_market(expected_goal_diff, handicap, home_odds, away_odds)
    
    model_home = market.implied_probs["home"]
    model_away = market.implied_probs["away"]
    
    edge_home = model_home - market.implied_probs["home"]
    edge_away = model_away - market.implied_probs["away"]
    
    if edge_home > min_edge:
        kelly = (model_home * home_odds - 1) / (home_odds - 1)
        return BetOpportunity(
            market_type="asian_handicap",
            outcome="home",
            model_prob=model_home,
            market_prob=market.implied_probs["home"],
            odds=home_odds,
            edge=edge_home,
            kelly_stake=max(kelly, 0),
            confidence="high" if edge_home > 0.1 else "medium"
        )
    
    if edge_away > min_edge:
        kelly = (model_away * away_odds - 1) / (away_odds - 1)
        return BetOpportunity(
            market_type="asian_handicap",
            outcome="away",
            model_prob=model_away,
            market_prob=market.implied_probs["away"],
            odds=away_odds,
            edge=edge_away,
            kelly_stake=max(kelly, 0),
            confidence="high" if edge_away > 0.1 else "medium"
        )
    
    return None


# ======================================================================
# Multi-Market Scanner
# ======================================================================
def scan_all_markets(
    expected_home_goals: float,
    expected_away_goals: float,
    match_odds: Dict[str, float],
    totals_lines: List[float] = None,
    handicap_lines: List[float] = None,
    min_edge: float = 0.05
) -> List[BetOpportunity]:
    """Scan all markets for edge.
    
    Args:
        expected_home_goals: Model's expected home goals
        expected_away_goals: Model's expected away goals
        match_odds: Match result odds {"home": 2.0, "draw": 3.5, "away": 3.8}
        totals_lines: List of totals lines to check
        handicap_lines: List of handicap lines to check
        min_edge: Minimum edge to flag
    
    Returns:
        List of BetOpportunity objects
    """
    opportunities = []
    
    expected_total = expected_home_goals + expected_away_goals
    expected_diff = expected_home_goals - expected_away_goals
    
    # Check totals markets
    if totals_lines is None:
        totals_lines = [1.5, 2.5, 3.5, 4.5]
    
    for line in totals_lines:
        # Typical odds for totals
        over_odds = 1.85
        under_odds = 1.95
        
        opp = totals_edge(expected_total, line, over_odds, under_odds, min_edge)
        if opp:
            opportunities.append(opp)
    
    # Check Asian Handicap markets
    if handicap_lines is None:
        handicap_lines = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
    
    for handicap in handicap_lines:
        home_odds = 1.9
        away_odds = 1.9
        
        opp = asian_handicap_edge(expected_diff, handicap, home_odds, away_odds, min_edge)
        if opp:
            opportunities.append(opp)
    
    # Sort by edge (highest first)
    opportunities.sort(key=lambda x: x.edge, reverse=True)
    
    return opportunities


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    print("Testing alternative markets module...")
    
    # Test totals
    total_market = totals_market(
        expected_total_goals=2.7,
        line=2.5,
        over_odds=1.85,
        under_odds=1.95
    )
    print(f"  Totals market: margin={total_market.margin:.3f}")
    print(f"    Model: over={total_market.implied_probs['over']:.3f}, under={total_market.implied_probs['under']:.3f}")
    
    # Test Asian Handicap
    ah_market = asian_handicap_market(
        expected_goal_diff=0.8,
        handicap=-1.0,
        home_odds=1.9,
        away_odds=1.9
    )
    print(f"  Asian Handicap: margin={ah_market.margin:.3f}")
    print(f"    Model: home={ah_market.implied_probs['home']:.3f}, away={ah_market.implied_probs['away']:.3f}")
    
    # Test multi-market scan
    opportunities = scan_all_markets(
        expected_home_goals=1.8,
        expected_away_goals=1.2,
        match_odds={"home": 2.0, "draw": 3.5, "away": 3.8}
    )
    print(f"\n  Found {len(opportunities)} opportunities across all markets")
    for opp in opportunities[:5]:
        print(f"    {opp.market_type}: {opp.outcome} (edge={opp.edge:.3f})")
    
    print("\n[OK] Alternative markets module complete.")
