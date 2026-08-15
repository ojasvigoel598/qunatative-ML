#!/usr/bin/env python3
"""
The chronological ML betting agent.

Wraps `AdaptiveMatchPredictor` (league-agnostic PoissonElo + Gradient Boosting
with online Elo/form updates and scheduled/drift refits) and adds the
decision-time behaviour required of a real-time agent:

  * PER-LEAGUE LEARNED TRUST — the agent tracks each league's rolling betting
    ROI and raises the edge threshold for leagues that have been LOSING it
    money (learned only from that league's own past bets, never hindsight),
    and demands a larger edge for leagues it has barely seen (familiarity).
  * FATIGUE — rest days since each team last played (known from revealed
    fixtures only).
  * STAKING — flat $10K (variance-minimising default) or quarter-Kelly, with a
    15% daily exposure cap and a hard 2% of bankroll cap per bet.
  * SURVIVAL MODE — if the bankroll drops below 10% of the start, switch to
    tiny flat stakes (0.5% of bankroll) and a relaxed edge threshold, spread
    across many matches, until recovery.

Every decision records a full trace (probs, odds, edges, stake, reason) so the
simulation ledger is auditable.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from models.adaptive_model import AdaptiveMatchPredictor

RESULT_TO_OUTCOME = {"H": "home_win", "D": "draw", "A": "away_win"}
OUTCOME_TO_RESULT = {"home_win": "H", "draw": "D", "away_win": "A"}

EDGE_BASE = 0.03
PROB_FLOOR = 0.40
MIN_ODDS = 1.6
FLAT_STAKE = 10_000.0
KELLY_FRACTION = 0.25
STAKE_CAP_FRAC = 0.02
DAILY_CAP_FRAC = 0.15
SURVIVAL_FLOOR = 0.10
SURVIVAL_EDGE = 0.01
SURVIVAL_PROB_FLOOR = 0.30
SURVIVAL_STAKE_FRAC = 0.005
LEAGUE_WINDOW = 20          # bets used for per-league rolling ROI
LEAGUE_PENALTY_SCALE = 0.5  # how strongly losing leagues raise their threshold
LEAGUE_FAMILIARITY_MIN = 30  # matches seen before a league is "known"
FAMILIARITY_PENALTY = 0.015


class BettingAgent:
    """Real-time agent over one world: predicts, bets, learns, never sees the future."""

    def __init__(self, train_df: pd.DataFrame, bankroll: float = 1_000_000.0,
                 stake_mode: str = "flat", seed: int = 42, base_kwargs: dict = None,
                 model_kwargs: dict = None):
        self.start = float(bankroll)
        self.bankroll = self.start
        self.peak = self.start
        self.stake_mode = stake_mode
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        kwargs = {"window": 240, "refit_every": 60, "drift_tol": 0.02,
                  "min_refit": 40, "seed": self.seed}
        if base_kwargs:
            kwargs.update(base_kwargs)
        self.model = AdaptiveMatchPredictor(**kwargs)
        if len(train_df):
            self.model.train(train_df)

        self.last_played = {}                 # team -> last known match date
        self.league_roi = {lg: deque(maxlen=LEAGUE_WINDOW)
                           for lg in set(train_df.get("league_code", []))}
        self.league_seen = {}                 # league_code -> matches revealed
        self.league_bets = {}                 # league_code -> n bets
        self.survival = False
        self.survival_day = None
        self.n_bets = 0
        self.n_wins = 0
        self.total_staked = 0.0
        self.daily_used = 0.0
        self.last_day = None
        self.last_known_date = None           # max date of data the agent has
        self.trace = []

    # ------------------------------------------------------------ knowledge
    def reveal_result(self, match: pd.Series):
        """Learn a result AFTER it happened (only from revealed leagues)."""
        lg = match["league_code"]
        self.league_seen[lg] = self.league_seen.get(lg, 0) + 1
        self.last_played[match["home_team"]] = match["date"]
        self.last_played[match["away_team"]] = match["date"]
        self.last_known_date = max(self.last_known_date or match["date"],
                                   match["date"])
        self.model.observe(match["home_team"], match["away_team"],
                           float(match["home_goals"]), float(match["away_goals"]),
                           match["result"])

    # ------------------------------------------------------------- decision
    def decide(self, match: pd.Series, sim_day, bankroll_before: float) -> dict:
        """Decide bet/no-bet for one upcoming match using only known info."""
        lg = match["league_code"]
        home, away = match["home_team"], match["away_team"]

        # fatigue (rest days) — only from revealed fixtures
        rest_h = max(0, (sim_day - self.last_played.get(
            home, sim_day - pd.Timedelta(days=6))).days)
        rest_a = max(0, (sim_day - self.last_played.get(
            away, sim_day - pd.Timedelta(days=6))).days)

        # model probabilities (Elo/form contain ONLY matches before today)
        p = self.model.predict(home, away)
        probs = {"home_win": float(p["home_win"]),
                 "draw": float(p["draw"]),
                 "away_win": float(p["away_win"])}

        # public price = B365 (what you can get), sharp = Pinnacle (CLV ref)
        odds = {"home_win": float(match.get("odds_home", np.nan)),
                "draw": float(match.get("odds_draw", np.nan)),
                "away_win": float(match.get("odds_away", np.nan))}
        sharp = {"home_win": float(match.get("pin_home", np.nan)),
                 "draw": float(match.get("pin_draw", np.nan)),
                 "away_win": float(match.get("pin_away", np.nan))}

        # per-league edge threshold (chronological, learned from its own bets)
        roll = list(self.league_roi.get(lg, []))
        rolling_roi = float(np.mean(roll)) if roll else 0.0
        penalty = max(0.0, -rolling_roi) * LEAGUE_PENALTY_SCALE
        if self.league_seen.get(lg, 0) < LEAGUE_FAMILIARITY_MIN:
            penalty += FAMILIARITY_PENALTY
        threshold = SURVIVAL_EDGE if self.survival else EDGE_BASE + penalty

        edges = {k: (probs[k] * odds[k] - 1.0) if pd.notna(odds[k]) and odds[k] > 1.0
                 else -1.0 for k in probs}
        best = max(edges, key=edges.get)
        edge = edges[best]
        prob_floor = SURVIVAL_PROB_FLOOR if self.survival else PROB_FLOOR
        conf = float(np.clip((max(probs.values()) - 1 / 3) / (2 / 3), 0.0, 1.0))

        reason, stake, decision = "no edge", 0.0, None
        if edge > threshold:
            if probs[best] >= prob_floor and odds[best] >= MIN_ODDS:
                if self.survival:
                    stake = SURVIVAL_STAKE_FRAC * self.bankroll
                elif self.stake_mode == "kelly":
                    kelly = max(0.0, edge / (odds[best] - 1.0))
                    stake = min(KELLY_FRACTION * kelly * self.bankroll,
                                STAKE_CAP_FRAC * self.bankroll)
                else:
                    stake = min(FLAT_STAKE, STAKE_CAP_FRAC * self.bankroll)
                stake = min(stake, max(0.0, DAILY_CAP_FRAC * self.bankroll
                                       - self.daily_used))
                if stake >= 1.0:
                    decision = best
                    reason = (f"edge {edge:+.1%} > {threshold:+.1%}"
                              f"{f' (league penalty {penalty:+.1%})' if penalty else ''}"
                              f"{', survival' if self.survival else ''}")
                else:
                    reason = "stake below minimum (daily cap)"
            else:
                reason = (f"prob {probs[best]:.0%} < floor {prob_floor:.0%}"
                          if probs[best] < prob_floor
                          else f"odds {odds[best]:.2f} < {MIN_ODDS}")
        elif edge <= 0:
            reason = "no positive edge"

        trace = {
            "match": f"{home} vs {away}", "league": match["league"],
            "league_code": lg, "sim_day": sim_day, "kickoff": match["date"],
            "home": home, "away": away,
            "probs": {k: round(v, 4) for k, v in probs.items()},
            "odds": {k: (round(v, 2) if pd.notna(v) else None) for k, v in odds.items()},
            "sharp": {k: (round(v, 2) if pd.notna(v) else None) for k, v in sharp.items()},
            "edges": {k: round(v, 4) for k, v in edges.items()},
            "best": best, "edge": round(edge, 4), "threshold": round(threshold, 4),
            "confidence": round(conf, 3), "rest_days": (rest_h, rest_a),
            "rolling_league_roi": round(rolling_roi, 4),
            "league_seen": self.league_seen.get(lg, 0),
            "decision": decision, "reason": reason, "stake": round(stake, 2),
            "survival": self.survival,
        }
        self.trace.append(trace)
        return trace

    # --------------------------------------------------------------- settle
    def settle(self, decision: dict, match: pd.Series, bankroll_before: float):
        """Settle a placed bet after the real result is revealed.

        `decision` may be either the raw trace from `decide()` (which carries
        an ``odds`` dict) or the flattened ledger row the engine passes back
        (``odds_home`` / ``odds_draw`` / ``odds_away``) — both shapes work.
        """
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
        self.daily_used += stake
        self.n_bets += 1
        self.n_wins += int(won)
        self.total_staked += stake
        lg = decision["league_code"]
        self.league_roi.setdefault(lg, deque(maxlen=LEAGUE_WINDOW)).append(
            profit / stake)
        self.league_bets[lg] = self.league_bets.get(lg, 0) + 1
        if self.bankroll < SURVIVAL_FLOOR * self.start and not self.survival:
            self.survival = True
            self.survival_day = match["date"]
        return self.bankroll, profit

    def end_day(self, sim_day):
        if self.last_day is not None and sim_day != self.last_day:
            self.daily_used = 0.0
        self.last_day = sim_day

    # -------------------------------------------------------------- summary
    def summary(self) -> dict:
        return {
            "final_bankroll": self.bankroll,
            "roi_pct": (self.bankroll / self.start - 1) * 100,
            "n_bets": self.n_bets, "n_wins": self.n_wins,
            "strike_rate": self.n_wins / self.n_bets if self.n_bets else 0.0,
            "total_staked": self.total_staked,
            "survival": self.survival,
            "survival_day": self.survival_day,
            "model_refits": getattr(self.model, "refits", 0),
            "leagues_bet": dict(self.league_bets),
        }

    def freeze_for_validation(self):
        """Freeze the ML layer (no more refits) for the untouched final window."""
        self.model.freeze()
