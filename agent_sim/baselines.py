#!/usr/bin/env python3
"""
Baseline policies for the multi-league agent simulation.

Each baseline walks the SAME world through the SAME chronological engine, so
the comparison is apples-to-apples.  Baselines do not learn (no Elo/form/refit)
— they are the reference points the ML agent has to beat (req 12):

    nobet   — never bet (benchmark of doing nothing)
    random  — bet a uniformly random outcome at a flat stake (no signal)
    implied — bet the outcome the BOOKMAKER prices most likely (implied prob
              >= 0.40) at a flat stake (market-follower, no model)
    fixed   — bet the max-edge outcome at a flat stake with NO edge threshold
              (pure fixed-stake baseline)
    ml      — the BettingAgent itself (the "existing model")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RESULT_TO_OUTCOME = {"H": "home_win", "D": "draw", "A": "away_win"}
FLAT_STAKE = 10_000.0
IMPLIED_FLOOR = 0.40


def _clean_odds(match) -> dict:
    return {"home_win": float(match.get("odds_home", np.nan)),
            "draw": float(match.get("odds_draw", np.nan)),
            "away_win": float(match.get("odds_away", np.nan))}


def implied_probs(odds: dict) -> dict:
    inv = {k: 1.0 / v for k, v in odds.items() if pd.notna(v) and v > 1.0}
    tot = sum(inv.values())
    return {k: v / tot for k, v in inv.items()} if tot > 0 else {}


class BaselinePolicy:
    """Minimal policy with the same interface the engine expects."""

    def __init__(self, mode: str, seed: int, bankroll: float = 1_000_000.0):
        self.mode = mode
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.start = float(bankroll)
        self.bankroll = self.start
        self.peak = self.start
        self.n_bets = 0
        self.n_wins = 0
        self.total_staked = 0.0
        self.last_known_date = None
        self.daily_used = 0.0
        self.last_day = None
        self.survival = False
        self.survival_day = None
        self.frozen = False

    # ---- engine interface -------------------------------------------------
    def decide(self, match: pd.Series, sim_day, bankroll_before: float) -> dict:
        odds = _clean_odds(match)
        ip = implied_probs(odds)
        outcome = None
        reason = "no-bet baseline"
        stake = 0.0
        edge = 0.0
        if self.mode == "random" and ip:
            outcome = str(self.rng.choice(list(ip.keys())))
            reason = "random"
            stake = FLAT_STAKE
            edge = ip[outcome] * odds[outcome] - 1.0
        elif self.mode == "implied" and ip:
            best = max(ip, key=ip.get)
            if ip[best] >= IMPLIED_FLOOR and odds[best] > 1.0:
                outcome = best
                reason = f"implied {ip[best]:.0%} >= {IMPLIED_FLOOR:.0%}"
                stake = FLAT_STAKE
                edge = ip[best] * odds[best] - 1.0
            else:
                reason = f"implied {ip.get(best, 0):.0%} < {IMPLIED_FLOOR:.0%}"
        elif self.mode == "fixed" and ip:
            best = max(ip, key=ip.get)
            if odds[best] > 1.0:
                outcome = best
                reason = "fixed-stake"
                stake = FLAT_STAKE
                edge = ip[best] * odds[best] - 1.0
        return {
            "match": f"{match['home_team']} vs {match['away_team']}",
            "league": match["league"], "league_code": match["league_code"],
            "kickoff": match["date"], "home": match["home_team"],
            "away": match["away_team"],
            "probs": {k: round(ip.get(k, 0.333), 4) for k in
                      ("home_win", "draw", "away_win")},
            "odds": {k: (round(v, 2) if pd.notna(v) else None)
                     for k, v in odds.items()},
            "sharp": {"home_win": None, "draw": None, "away_win": None},
            "edges": {k: round(ip.get(k, 0.333) * (odds.get(k) or 1) - 1, 4)
                      for k in ("home_win", "draw", "away_win")},
            "best": outcome, "edge": round(edge, 4), "threshold": 0.0,
            "confidence": 0.0, "rest_days": (0, 0),
            "rolling_league_roi": 0.0, "league_seen": 0,
            "decision": outcome, "reason": reason, "stake": round(stake, 2),
            "survival": False,
        }

    def settle(self, decision: dict, match: pd.Series, bankroll_before: float):
        if decision["decision"] is None:
            return bankroll_before, None
        outcome = decision["decision"]
        odds_map = decision.get("odds")
        if odds_map is None or odds_map.get(outcome) is None:
            odds_map = {"home_win": decision.get("odds_home"),
                        "draw": decision.get("odds_draw"),
                        "away_win": decision.get("odds_away")}
        odds = odds_map.get(outcome) or 0.0
        if not odds or odds <= 1.0:
            # a decided bet must have valid odds — never settle a phantom bet
            return bankroll_before, None
        stake = decision["stake"]
        won = RESULT_TO_OUTCOME[match["result"]] == outcome
        profit = stake * (odds - 1.0) if won else -stake
        self.bankroll += profit
        self.peak = max(self.peak, self.bankroll)
        self.n_bets += 1
        self.n_wins += int(won)
        self.total_staked += stake
        return self.bankroll, profit

    def reveal_result(self, match: pd.Series):
        pass

    def end_day(self, sim_day):
        if self.last_day is not None and sim_day != self.last_day:
            self.daily_used = 0.0
        self.last_day = sim_day

    def freeze_for_validation(self):
        self.frozen = True

    def summary(self) -> dict:
        return {
            "final_bankroll": self.bankroll,
            "roi_pct": (self.bankroll / self.start - 1) * 100,
            "n_bets": self.n_bets, "n_wins": self.n_wins,
            "strike_rate": self.n_wins / self.n_bets if self.n_bets else 0.0,
            "total_staked": self.total_staked,
            "survival": self.survival,
            "survival_day": self.survival_day,
            "model_refits": 0,
            "leagues_bet": {},
        }
