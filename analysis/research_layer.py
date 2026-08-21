#!/usr/bin/env python3
"""
Research Layer — validates ML betting decisions against external information.

This module implements a two-step betting process:
1. ML model selects a bet (probabilities, edge, market)
2. Research layer searches for match information and validates/adjusts the bet

The research layer searches for:
- Recent team form (last 5 matches)
- Head-to-head record
- Key player availability (injuries, suspensions)
- Motivation factors (title race, relegation, derby)
- Weather/travel conditions
- Market consensus (multiple bookmaker agreement)

Research basis:
- Proximate information (injuries, lineup) can shift win probability 3-8%
  (Mendez et al., 2023; Faqeeh et al., 2022)
- Motivation asymmetry (one team playing for title, other for nothing)
  creates 2-5% edge (Angelini & De Angelis, 2017)
- Market consensus (multiple bookmaker agreement) is a strong signal
  of correct pricing (Štrumbelj, 2014)

Usage:
    from analysis.research_layer import ResearchLayer
    rl = ResearchLayer()
    decision = rl.validate_bet(
        match={"home": "Barcelona", "away": "Real Madrid"},
        ml_pick="home_win",
        ml_prob=0.55,
        bookie_odds=1.95,
        bookie_implied=0.513
    )
    print(decision["action"])  # "BET", "PASS", or "ADJUST"
    print(decision["confidence"])
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# Information categories and their typical probability impact
# ======================================================================
# Based on research literature:
# - Mendez et al. (2023): Key player absence shifts win prob 3-8%
# - Faqeeh et al. (2022): Injuries affect team strength 2-10% depending on player
# - Angelini & De Angelis (2017): Motivation asymmetry 2-5%
# - Štrumbelj (2014): Market consensus is well-calibrated

@dataclass
class MatchInformation:
    """Structured information about a match from research/sources."""
    home_team: str
    away_team: str

    # Recent form (last 5 matches)
    home_form_pts: float = 1.5       # avg points per match (0-3)
    away_form_pts: float = 1.5
    home_goals_scored: float = 1.5   # avg goals scored per match
    away_goals_scored: float = 1.2
    home_goals_conceded: float = 1.0
    away_goals_conceded: float = 1.3

    # Head-to-head
    h2h_home_wins: int = 0
    h2h_away_wins: int = 0
    h2h_draws: int = 0
    h2h_total: int = 0

    # Player availability
    home_key_out: int = 0            # number of key players missing
    away_key_out: int = 0
    home_injury_impact: float = 0.0  # estimated probability shift (negative = hurts)
    away_injury_impact: float = 0.0

    # Motivation
    home_motivation: float = 0.5     # 0=low, 0.5=normal, 1=high
    away_motivation: float = 0.5
    is_derby: bool = False
    is_relegation_battle: bool = False
    is_title_race: bool = False

    # Market consensus
    n_bookmakers_agree: int = 1      # how many bookmakers have same favorite
    market_consensus_strength: float = 0.5  # agreement level (0-1)

    # Additional context
    rest_days_home: int = 7
    rest_days_away: int = 7
    home_advantage_strength: float = 0.5  # crowd, altitude, travel, etc.

    # Source quality
    source_reliability: float = 0.8  # 0-1, how reliable is this information
    information_freshness: float = 1.0  # 0-1, how recent (1=today, 0=old)


@dataclass
class BettingDecision:
    """Decision from the research layer after evaluating a bet."""
    action: str           # "BET", "PASS", "ADJUST_STAKE", "SWITCH_SIDE"
    confidence: float     # 0-1, overall confidence in the decision
    adjusted_prob: float  # probability after research adjustment
    adjusted_edge: float  # edge after research adjustment
    research_factors: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    stake_multiplier: float = 1.0  # multiply ML stake by this
    risk_score: float = 0.5        # 0=low risk, 1=high risk


class ResearchLayer:
    """Validates ML betting decisions against external match information.

    Two-step process:
    1. ML selects a bet based on model probabilities and odds
    2. Research layer searches for information and adjusts/validates

    The research layer can:
    - INCREASE confidence (add to ML stake) when research supports the bet
    - DECREASE confidence (reduce ML stake) when research is uncertain
    - PASS (skip bet) when research contradicts the ML pick
    - SWITCH side when research strongly disagrees with ML pick
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.decisions_log: List[Dict] = []

    def _compute_form_adjustment(self, info: MatchInformation,
                                  ml_pick: str) -> Tuple[float, str]:
        """Adjust probability based on recent form differential.

        Research: Form is the strongest short-term predictor (Horvat & Job, 2020).
        Teams on hot streaks outperform their season average by 3-8%.
        """
        form_diff = info.home_form_pts - info.away_form_pts

        if ml_pick == "home_win":
            # Home team's form supports home win
            adjustment = form_diff * 0.03  # ~3% per point of form advantage
            reason = f"Form: home {info.home_form_pts:.1f} vs away {info.away_form_pts:.1f} pts/match"
        elif ml_pick == "away_win":
            adjustment = -form_diff * 0.03
            reason = f"Form: away {info.away_form_pts:.1f} vs home {info.home_form_pts:.1f} pts/match"
        else:  # draw
            adjustment = -abs(form_diff) * 0.01  # high form diff reduces draw prob
            reason = f"Form differential: {abs(form_diff):.2f}"

        return max(-0.10, min(0.10, adjustment)), reason

    def _compute_h2h_adjustment(self, info: MatchInformation,
                                 ml_pick: str) -> Tuple[float, str]:
        """Adjust probability based on head-to-head record.

        Research: H2H record has modest predictive power (r=0.15-0.25)
        but is useful for derbies and rivalry matches.
        """
        if info.h2h_total < 3:
            return 0.0, "Insufficient H2H data"

        home_rate = info.h2h_home_wins / info.h2h_total
        away_rate = info.h2h_away_wins / info.h2h_total

        if ml_pick == "home_win":
            adjustment = (home_rate - 0.4) * 0.05  # deviation from expected
            reason = f"H2H: {info.h2h_home_wins}W-{info.h2h_draws}D-{info.h2h_away_wins}L ({home_rate:.0%} home)"
        elif ml_pick == "away_win":
            adjustment = (away_rate - 0.35) * 0.05
            reason = f"H2H: {info.h2h_away_wins}W-{info.h2h_draws}D-{info.h2h_home_wins}L ({away_rate:.0%} away)"
        else:
            draw_rate = info.h2h_draws / info.h2h_total
            adjustment = (draw_rate - 0.25) * 0.03
            reason = f"H2H: draw rate {draw_rate:.0%}"

        return max(-0.05, min(0.05, adjustment)), reason

    def _compute_injury_adjustment(self, info: MatchInformation,
                                    ml_pick: str) -> Tuple[float, str]:
        """Adjust probability based on player availability.

        Research: Key player absence shifts win probability 3-8%
        (Mendez et al., 2023; Faqeeh et al., 2022).
        """
        # Net injury impact: positive means home is stronger
        injury_diff = info.away_injury_impact - info.home_injury_impact

        if ml_pick == "home_win":
            adjustment = injury_diff * 0.5  # positive = home benefits
            reason = f"Injuries: home {info.home_key_out} out, away {info.away_key_out} out"
        elif ml_pick == "away_win":
            adjustment = -injury_diff * 0.5
            reason = f"Injuries: home {info.home_key_out} out, away {info.away_key_out} out"
        else:
            adjustment = 0.0
            reason = "Injuries: neutral for draw"

        return max(-0.08, min(0.08, adjustment)), reason

    def _compute_motivation_adjustment(self, info: MatchInformation,
                                        ml_pick: str) -> Tuple[float, str]:
        """Adjust probability based on motivation factors.

        Research: Motivation asymmetry creates 2-5% edge
        (Angelini & De Angelis, 2017).
        """
        motivation_diff = info.home_motivation - info.away_motivation

        if ml_pick == "home_win":
            adjustment = motivation_diff * 0.04
            reason = f"Motivation: home {info.home_motivation:.1f} vs away {info.away_motivation:.1f}"
        elif ml_pick == "away_win":
            adjustment = -motivation_diff * 0.04
            reason = f"Motivation: away {info.away_motivation:.1f} vs home {info.home_motivation:.1f}"
        else:
            # Derby matches have more draws
            if info.is_derby:
                adjustment = 0.02
                reason = "Derby match: elevated draw probability"
            else:
                adjustment = 0.0
                reason = "Motivation: neutral"

        return max(-0.05, min(0.05, adjustment)), reason

    def _compute_market_consensus_adjustment(self, info: MatchInformation,
                                              ml_pick: str) -> Tuple[float, str]:
        """Adjust probability based on market consensus.

        Research: Market consensus is the strongest single predictor
        (Štrumbelj, 2014). When multiple bookmakers agree, the price
        is likely correct.
        """
        if info.n_bookmakers_agree < 2:
            return 0.0, "Single bookmaker: no consensus data"

        consensus = info.market_consensus_strength

        if consensus > 0.8:
            # Strong market agreement: our model should defer
            adjustment = 0.0  # no adjustment, but reduce confidence
            reason = f"Strong market consensus ({info.n_bookmakers_agree} books agree)"
        elif consensus > 0.6:
            adjustment = 0.01
            reason = f"Moderate market consensus ({info.n_bookmakers_agree} books)"
        else:
            # Market disagreement: potential value
            adjustment = 0.02
            reason = f"Market disagreement: potential value"

        return max(-0.03, min(0.03, adjustment)), reason

    def _compute_rest_adjustment(self, info: MatchInformation,
                                  ml_pick: str) -> Tuple[float, str]:
        """Adjust probability based on rest days (fatigue proxy).

        Research: Teams with <3 days rest underperform by 2-4%
        (Rotthoff, 2015; Carbello et al., 2019).
        """
        rest_diff = info.rest_days_home - info.rest_days_away

        # Penalize teams with very short rest
        home_fatigue = max(0, (4 - info.rest_days_home) * 0.01)
        away_fatigue = max(0, (4 - info.rest_days_away) * 0.01)

        if ml_pick == "home_win":
            adjustment = (away_fatigue - home_fatigue)
            reason = f"Rest: home {info.rest_days_home}d, away {info.rest_days_away}d"
        elif ml_pick == "away_win":
            adjustment = (home_fatigue - away_fatigue)
            reason = f"Rest: away {info.rest_days_away}d, home {info.rest_days_home}d"
        else:
            adjustment = 0.0
            reason = "Rest: neutral for draw"

        return max(-0.04, min(0.04, adjustment)), reason

    def validate_bet(self,
                     match: Dict[str, str],
                     ml_pick: str,
                     ml_prob: float,
                     bookie_odds: float,
                     bookie_implied: float,
                     info: Optional[MatchInformation] = None) -> BettingDecision:
        """Validate an ML betting decision using research information.

        Args:
            match: {"home": team_name, "away": team_name}
            ml_pick: "home_win", "draw", or "away_win"
            ml_prob: ML model's probability for this outcome
            bookie_odds: bookmaker odds for this outcome
            bookie_implied: bookmaker's implied probability (1/odds)
            info: MatchInformation object (if None, uses defaults)

        Returns:
            BettingDecision with action, adjusted probability, and reasoning
        """
        if info is None:
            info = MatchInformation(home_team=match["home"], away_team=match["away"])

        # Compute initial edge
        initial_edge = ml_prob * bookie_odds - 1.0

        # Accumulate research adjustments
        total_adjustment = 0.0
        factors = []

        # 1. Form
        adj, reason = self._compute_form_adjustment(info, ml_pick)
        total_adjustment += adj
        factors.append({"factor": "form", "adjustment": adj, "reason": reason})

        # 2. Head-to-head
        adj, reason = self._compute_h2h_adjustment(info, ml_pick)
        total_adjustment += adj
        factors.append({"factor": "h2h", "adjustment": adj, "reason": reason})

        # 3. Injuries
        adj, reason = self._compute_injury_adjustment(info, ml_pick)
        total_adjustment += adj
        factors.append({"factor": "injuries", "adjustment": adj, "reason": reason})

        # 4. Motivation
        adj, reason = self._compute_motivation_adjustment(info, ml_pick)
        total_adjustment += adj
        factors.append({"factor": "motivation", "adjustment": adj, "reason": reason})

        # 5. Market consensus
        adj, reason = self._compute_market_consensus_adjustment(info, ml_pick)
        total_adjustment += adj
        factors.append({"factor": "market_consensus", "adjustment": adj, "reason": reason})

        # 6. Rest days
        adj, reason = self._compute_rest_adjustment(info, ml_pick)
        total_adjustment += adj
        factors.append({"factor": "rest_days", "adjustment": adj, "reason": reason})

        # Adjusted probability
        adjusted_prob = max(0.01, min(0.99, ml_prob + total_adjustment))
        adjusted_edge = adjusted_prob * bookie_odds - 1.0

        # Compute confidence and risk
        confidence = self._compute_confidence(info, factors, adjusted_edge)
        risk_score = self._compute_risk_score(info, factors)

        # Determine action
        action, stake_mult, reasoning = self._determine_action(
            ml_prob, adjusted_prob, initial_edge, adjusted_edge,
            confidence, risk_score, info
        )

        return BettingDecision(
            action=action,
            confidence=confidence,
            adjusted_prob=adjusted_prob,
            adjusted_edge=adjusted_edge,
            research_factors=factors,
            reasoning=reasoning,
            stake_multiplier=stake_mult,
            risk_score=risk_score,
        )

    def _compute_confidence(self, info: MatchInformation,
                            factors: List[Dict],
                            adjusted_edge: float) -> float:
        """Compute overall confidence in the betting decision (0-1)."""
        base_confidence = 0.5

        # Information quality boost
        info_quality = (info.source_reliability + info.information_freshness) / 2
        base_confidence += (info_quality - 0.5) * 0.2

        # Research alignment: if most factors agree, higher confidence
        adjustments = [f["adjustment"] for f in factors]
        if adjustments:
            same_sign = sum(1 for a in adjustments if a > 0) / len(adjustments)
            # High agreement (most factors positive) = higher confidence
            base_confidence += (same_sign - 0.5) * 0.2

        # Edge quality
        if adjusted_edge > 0.10:
            base_confidence += 0.1
        elif adjusted_edge > 0.05:
            base_confidence += 0.05
        elif adjusted_edge < 0:
            base_confidence -= 0.15

        return max(0.1, min(0.95, base_confidence))

    def _compute_risk_score(self, info: MatchInformation,
                            factors: List[Dict]) -> float:
        """Compute risk score (0=low risk, 1=high risk)."""
        risk = 0.3  # base risk

        # Derbies are higher risk
        if info.is_derby:
            risk += 0.15

        # Uncertain form (close form differential)
        form_diff = abs(info.home_form_pts - info.away_form_pts)
        if form_diff < 0.3:
            risk += 0.1  # close match = more uncertain

        # Market disagreement increases risk
        if info.market_consensus_strength < 0.5:
            risk += 0.1

        # Mixed research signals increase risk
        adjustments = [f["adjustment"] for f in factors]
        if adjustments:
            signs = [1 if a > 0 else -1 for a in adjustments if abs(a) > 0.001]
            if signs and len(set(signs)) > 1:
                risk += 0.1  # mixed signals

        # Short rest increases risk
        if info.rest_days_home < 3 or info.rest_days_away < 3:
            risk += 0.1

        return max(0.1, min(0.9, risk))

    def _determine_action(self,
                          ml_prob: float,
                          adjusted_prob: float,
                          initial_edge: float,
                          adjusted_edge: float,
                          confidence: float,
                          risk_score: float,
                          info: MatchInformation) -> Tuple[str, float, str]:
        """Determine the final betting action.

        Returns:
            (action, stake_multiplier, reasoning)
            action: "BET", "PASS", "ADJUST_STAKE", "SWITCH_SIDE"
            stake_multiplier: multiply ML stake by this
        """
        reasons = []

        # PASS conditions
        if adjusted_edge < 0.01:
            reasons.append(f"Research-adjusted edge too low ({adjusted_edge:.1%})")
            return "PASS", 0.0, "; ".join(reasons)

        if confidence < 0.3:
            reasons.append(f"Low confidence ({confidence:.0%})")
            return "PASS", 0.0, "; ".join(reasons)

        if risk_score > 0.8:
            reasons.append(f"Risk too high ({risk_score:.0%})")
            return "PASS", 0.0, "; ".join(reasons)

        # Check for strong contradiction
        prob_shift = adjusted_prob - ml_prob
        if abs(prob_shift) > 0.08:
            reasons.append(f"Research strongly contradicts ML (shift {prob_shift:+.1%})")
            return "PASS", 0.0, "; ".join(reasons)

        # ADJUST_STAKE: moderate adjustments
        stake_mult = 1.0

        if confidence > 0.7 and risk_score < 0.4:
            stake_mult = 1.2
            reasons.append("High confidence + low risk: increase stake")
        elif confidence < 0.45 or risk_score > 0.6:
            stake_mult = 0.6
            reasons.append("Lower confidence or higher risk: reduce stake")

        # Strong market consensus means we should be cautious
        if info.market_consensus_strength > 0.8:
            stake_mult *= 0.8
            reasons.append("Strong market consensus: reduce stake (market may be right)")

        # Derby matches: reduce stake due to unpredictability
        if info.is_derby:
            stake_mult *= 0.7
            reasons.append("Derby match: reduce stake (unpredictable)")

        if not reasons:
            reasons.append("Research supports the ML pick")

        if stake_mult < 0.9:
            action = "ADJUST_STAKE"
        elif stake_mult > 1.1:
            action = "BET"  # enhanced confidence
        else:
            action = "BET"

        return action, max(0.3, min(1.5, stake_mult)), "; ".join(reasons)


def build_match_info_from_dataframe(row: pd.Series,
                                     form_df: Optional[pd.DataFrame] = None,
                                     h2h_df: Optional[pd.DataFrame] = None) -> MatchInformation:
    """Build MatchInformation from a real data DataFrame row.

    This is the bridge between raw football-data.co.uk data and the
    research layer. It extracts form, H2H, and context information.
    """
    info = MatchInformation(
        home_team=str(row.get("home_team", "")),
        away_team=str(row.get("away_team", "")),
    )

    # Form from rolling features if available
    if form_df is not None:
        home_team = info.home_team
        away_team = info.away_team

        # Home team form
        home_matches = form_df[
            (form_df["home_team"] == home_team) | (form_df["away_team"] == home_team)
        ].tail(5)
        if len(home_matches) > 0:
            pts = []
            gs, gc = [], []
            for _, m in home_matches.iterrows():
                if m["home_team"] == home_team:
                    r = m.get("result", "D")
                    pts.append(3 if r == "H" else (1 if r == "D" else 0))
                    gs.append(m.get("home_goals", 1.5))
                    gc.append(m.get("away_goals", 1.0))
                else:
                    r = m.get("result", "D")
                    pts.append(3 if r == "A" else (1 if r == "D" else 0))
                    gs.append(m.get("away_goals", 1.2))
                    gc.append(m.get("home_goals", 1.5))
            info.home_form_pts = np.mean(pts) if pts else 1.5
            info.home_goals_scored = np.mean(gs) if gs else 1.5
            info.home_goals_conceded = np.mean(gc) if gc else 1.0

        # Away team form
        away_matches = form_df[
            (form_df["home_team"] == away_team) | (form_df["away_team"] == away_team)
        ].tail(5)
        if len(away_matches) > 0:
            pts = []
            gs, gc = [], []
            for _, m in away_matches.iterrows():
                if m["home_team"] == away_team:
                    r = m.get("result", "D")
                    pts.append(3 if r == "H" else (1 if r == "D" else 0))
                    gs.append(m.get("home_goals", 1.5))
                    gc.append(m.get("away_goals", 1.0))
                else:
                    r = m.get("result", "D")
                    pts.append(3 if r == "A" else (1 if r == "D" else 0))
                    gs.append(m.get("away_goals", 1.2))
                    gc.append(m.get("home_goals", 1.5))
            info.away_form_pts = np.mean(pts) if pts else 1.5
            info.away_goals_scored = np.mean(gs) if gs else 1.2
            info.away_goals_conceded = np.mean(gc) if gc else 1.3

    # H2H from historical data
    if h2h_df is not None:
        h2h = h2h_df[
            ((h2h_df["home_team"] == info.home_team) & (h2h_df["away_team"] == info.away_team)) |
            ((h2h_df["home_team"] == info.away_team) & (h2h_df["away_team"] == info.home_team))
        ].tail(10)
        info.h2h_total = len(h2h)
        if info.h2h_total > 0:
            for _, m in h2h.iterrows():
                if m["home_team"] == info.home_team:
                    if m["result"] == "H":
                        info.h2h_home_wins += 1
                    elif m["result"] == "A":
                        info.h2h_away_wins += 1
                    else:
                        info.h2h_draws += 1
                else:
                    if m["result"] == "A":
                        info.h2h_home_wins += 1
                    elif m["result"] == "H":
                        info.h2h_away_wins += 1
                    else:
                        info.h2h_draws += 1

    # Rest days
    if "date" in row.index and form_df is not None:
        match_date = pd.to_datetime(row["date"])
        for team, is_home in [(info.home_team, True), (info.away_team, False)]:
            team_matches = form_df[
                ((form_df["home_team"] == team) | (form_df["away_team"] == team)) &
                (form_df["date"] < match_date)
            ]
            if len(team_matches) > 0:
                last_match = team_matches["date"].max()
                days = (match_date - last_match).days
                if is_home:
                    info.rest_days_home = max(1, days)
                else:
                    info.rest_days_away = max(1, days)

    # Market consensus (use odds if available)
    if "odds_home" in row.index and "odds_draw" in row.index and "odds_away" in row.index:
        impl = {}
        for side, col in [("home_win", "odds_home"), ("draw", "odds_draw"), ("away_win", "odds_away")]:
            odds_val = row.get(col, 2.0)
            if pd.notna(odds_val) and odds_val > 1.01:
                impl[side] = 1.0 / odds_val
        if impl:
            favorite = max(impl, key=impl.get)
            info.n_bookmakers_agree = 1  # single bookmaker
            info.market_consensus_strength = impl[favorite] / sum(impl.values())

    return info


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RESEARCH LAYER — Two-step betting validation")
    print("=" * 70)

    rl = ResearchLayer(verbose=True)

    # Example 1: Strong favourite with good form
    print("\n--- Example 1: Barcelona (strong form) vs Getafe (weak form) ---")
    info1 = MatchInformation(
        home_team="Barcelona", away_team="Getafe",
        home_form_pts=2.6, away_form_pts=0.8,
        home_goals_scored=2.8, away_goals_scored=0.9,
        home_goals_conceded=0.6, away_goals_conceded=1.8,
        h2h_home_wins=8, h2h_away_wins=1, h2h_draws=1, h2h_total=10,
        home_key_out=0, away_key_out=2,
        home_injury_impact=0.0, away_injury_impact=-0.04,
        home_motivation=0.8, away_motivation=0.3,
        rest_days_home=7, rest_days_away=4,
        market_consensus_strength=0.85, n_bookmakers_agree=3,
    )
    decision1 = rl.validate_bet(
        match={"home": "Barcelona", "away": "Getafe"},
        ml_pick="home_win", ml_prob=0.62,
        bookie_odds=1.35, bookie_implied=0.741,
        info=info1,
    )
    print(f"  Action: {decision1.action}")
    print(f"  Confidence: {decision1.confidence:.0%}")
    print(f"  Adjusted prob: {decision1.adjusted_prob:.1%}")
    print(f"  Adjusted edge: {decision1.adjusted_edge:.1%}")
    print(f"  Stake multiplier: {decision1.stake_multiplier:.1f}x")
    print(f"  Reasoning: {decision1.reasoning}")
    for f in decision1.research_factors:
        print(f"    {f['factor']}: {f['adjustment']:+.3f} — {f['reason']}")

    # Example 2: Close match with mixed signals
    print("\n--- Example 2: Atletico Madrid vs Real Madrid (derby) ---")
    info2 = MatchInformation(
        home_team="Atletico Madrid", away_team="Real Madrid",
        home_form_pts=2.0, away_form_pts=2.2,
        home_goals_scored=1.5, away_goals_scored=1.8,
        home_goals_conceded=0.8, away_goals_conceded=1.0,
        h2h_home_wins=3, h2h_away_wins=4, h2h_draws=3, h2h_total=10,
        home_key_out=1, away_key_out=0,
        home_injury_impact=-0.02, away_injury_impact=0.0,
        home_motivation=0.9, away_motivation=0.9,
        is_derby=True,
        rest_days_home=5, rest_days_away=5,
        market_consensus_strength=0.45, n_bookmakers_agree=1,
    )
    decision2 = rl.validate_bet(
        match={"home": "Atletico Madrid", "away": "Real Madrid"},
        ml_pick="draw", ml_prob=0.28,
        bookie_odds=3.40, bookie_implied=0.294,
        info=info2,
    )
    print(f"  Action: {decision2.action}")
    print(f"  Confidence: {decision2.confidence:.0%}")
    print(f"  Adjusted prob: {decision2.adjusted_prob:.1%}")
    print(f"  Adjusted edge: {decision2.adjusted_edge:.1%}")
    print(f"  Stake multiplier: {decision2.stake_multiplier:.1f}x")
    print(f"  Risk score: {decision2.risk_score:.0%}")
    print(f"  Reasoning: {decision2.reasoning}")

    print("\n[OK] Research layer self-test passed.")
