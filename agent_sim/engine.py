#!/usr/bin/env python3
"""
The chronological walk-forward engine.

For every simulation day, in order:

    RESOLVE  matches kicking off today (result revealed, bets settled,
             bankroll updated, model learns)   -> kickoff == sim_day
    OFFER    upcoming fixtures of REVEALED leagues with kickoff in
             (sim_day, sim_day + lookahead]     -> prediction time == sim_day

The engine enforces and AUDITS two invariants on every opportunity:

    feature_timestamp  <= prediction_timestamp     (data cutoff <= decision day)
    result_timestamp   >  prediction_timestamp     (kickoff   >  decision day)

If either is ever violated the opportunity is FLAGGED as data leakage and the
bet is INVALIDATED (excluded from the bankroll).  Violations are impossible
by construction of the walk, but the audit records every field so the
simulation can be verified after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent_sim.agent import BettingAgent
from agent_sim.ledger import RollingLedger


class SimulationEngine:
    def __init__(self, world, agent: BettingAgent, ledger: RollingLedger,
                 frozen_frac: float = 0.0):
        self.world = world
        self.agent = agent
        self.ledger = ledger
        self.frozen_frac = frozen_frac
        n = len(world.sim_dates)
        self.freeze_date = (world.sim_dates[int(n * (1 - frozen_frac))]
                            if frozen_frac > 0 else None)
        self.frozen_armed = False

        self.offered = set()                 # match ids already offered
        self.pending = {}                    # kickoff -> [(match_id, row_idx)]
        self.row_by_match = {}               # match_id -> ledger row index
        self.bankroll = agent.start
        self.n_leak_flags = 0

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        # pre-warm the agent's knowledge cutoff to its training max date
        if len(self.world.train_df):
            self.agent.last_known_date = self.world.train_df["date"].max()

        for sim_day in self.world.sim_dates:
            if self.freeze_date is not None and sim_day >= self.freeze_date \
                    and not self.frozen_armed:
                self.agent.freeze_for_validation()
                self.frozen_armed = True

            self._resolve(sim_day)      # kickoff == sim_day
            self._offer(sim_day)        # kickoff in (sim_day, +lookahead]
            self.agent.end_day(sim_day)

        summary = self.agent.summary()
        summary.update({
            "matches_evaluated": len(self.ledger.rows),
            "n_bets": self.agent.n_bets,
            "n_wins": self.agent.n_wins,
            "n_leak_flags": self.n_leak_flags,
            "frozen_window": self.frozen_armed,
        })
        return summary

    # ------------------------------------------------------------ resolution
    def _resolve(self, sim_day: pd.Timestamp):
        # pop the day's due bets ONCE — several matches may kick off on the
        # same day, and each must settle against its own ledger row.
        due = self.pending.pop(sim_day, [])
        for m in self.world.results_on(sim_day):
            mid = m.name
            # settle any bet placed on this match
            for mid2, row_idx in due:
                if mid2 != mid:
                    continue
                self._settle(row_idx, m)
                break
            # learn the result (only matches from revealed leagues, i.e. only
            # matches the agent could have known about — guaranteed, since a
            # match is only offered after its league reveals)
            self.agent.reveal_result(m)
            self.ledger.note_resolved()

    def _settle(self, row_idx: int, match: pd.Series):
        row = self.ledger.rows[row_idx]
        bankroll_before = row["bankroll_before"]
        if row["invalidated"]:
            # leak: the bet never counted
            row["profit"] = 0.0
            row["bankroll_after"] = bankroll_before
            return
        final_bankroll, profit = self.agent.settle(row, match, bankroll_before)
        row["result"] = match["result"]
        row["profit"] = round(profit, 2) if profit is not None else 0.0
        row["bankroll_after"] = round(final_bankroll, 2)

    # --------------------------------------------------------------- offers
    def _offer(self, sim_day: pd.Timestamp):
        for m in self.world.upcoming_matches(sim_day):
            mid = m.name
            if mid in self.offered:
                continue
            self.offered.add(mid)

            decision = self.agent.decide(m, sim_day, self.agent.bankroll)
            row = self._build_row(decision, sim_day)

            # ---- strict leakage audit (req 5) ---------------------------
            kickoff = m["date"]
            cutoff = decision.get("feature_cutoff") or (
                self.agent.last_known_date or sim_day)
            leak_feature = (pd.notna(cutoff) and cutoff > sim_day)
            leak_result = (kickoff <= sim_day)
            row["data_cutoff"] = cutoff.date() if pd.notna(cutoff) else ""
            row["result_known_at_prediction"] = bool(leak_result)
            row["leak_flag"] = int(leak_feature or leak_result)
            if row["leak_flag"]:
                self.n_leak_flags += 1
                row["invalidated"] = 1
                row["reason"] = "DATA LEAKAGE — prediction invalidated"
                # also drop any stake decision that slipped through
                row["stake"] = 0.0
                row["decision"] = None
            else:
                row["invalidated"] = 0

            if row["decision"] is not None:
                self.ledger.add(row)
                row_idx = len(self.ledger.rows) - 1   # index, not the dict
                self.pending.setdefault(kickoff, []).append((mid, row_idx))
                self.row_by_match[mid] = row_idx
            else:
                self.ledger.add(row)

    # ------------------------------------------------------------ row build
    def _build_row(self, d: dict, sim_day: pd.Timestamp) -> dict:
        return {
            "bet_no": None,   # assigned at settlement for bets
            "timestamp": sim_day,
            "kickoff": d["kickoff"],
            "data_cutoff": None,
            "league": d["league"], "league_code": d["league_code"],
            "match": d["match"], "home": d["home"], "away": d["away"],
            "prediction": d["best"] if d["decision"] else None,
            "prob_home": d["probs"]["home_win"],
            "prob_draw": d["probs"]["draw"],
            "prob_away": d["probs"]["away_win"],
            "odds_home": d["odds"]["home_win"],
            "odds_draw": d["odds"]["draw"],
            "odds_away": d["odds"]["away_win"],
            "sharp_home": d["sharp"]["home_win"],
            "sharp_draw": d["sharp"]["draw"],
            "sharp_away": d["sharp"]["away_win"],
            "edge": d["edge"], "threshold": d["threshold"],
            "confidence": d["confidence"],
            "rest_home": d["rest_days"][0], "rest_away": d["rest_days"][1],
            "league_roi": d["rolling_league_roi"],
            "league_seen": d["league_seen"],
            "decision": d["decision"], "reason": d["reason"],
            "stake": d["stake"], "result": None,
            "result_known_at_prediction": False,
            "leak_flag": 0, "invalidated": 0,
            # NOTE: bankroll_before is captured at OFFER time; bets settle at
            # kickoff, so ledger rows are not strictly contiguous. Metrics use
            # profit sums + the final bankroll, which stay consistent.
            "bankroll_before": round(self.agent.bankroll, 2),
            "profit": 0.0,
            "bankroll_after": round(self.agent.bankroll, 2),
        }

    # ----------------------------------------------------------- post-run
    def number_bets(self):
        """Assign bet numbers to settled bets (chronological order)."""
        n = 0
        for i, row in enumerate(self.ledger.rows):
            if row["decision"] is not None and not row["invalidated"]:
                n += 1
                row["bet_no"] = n

    def league_timeline(self) -> pd.DataFrame:
        """When each league was revealed and first bet (req 10)."""
        rows = []
        for lg in self.world.leagues:
            reveal = self.world.reveal[lg]
            first_bet = None
            for r in self.ledger.rows:
                if r["league_code"] == lg and r["decision"] is not None \
                        and not r["invalidated"]:
                    first_bet = r["timestamp"]
                    break
            rows.append({"league": lg, "league_name": self.world.league_names[lg],
                         "revealed": reveal, "first_bet": first_bet})
        return pd.DataFrame(rows)
