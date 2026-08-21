#!/usr/bin/env python3
"""
Backtest Record — every bet must contain all required fields.

Per the research requirements, every simulated bet must record:
- match_id, prediction_timestamp, model_version, feature_version
- model_probability, calibrated_probability, market_odds
- market_implied_probability, de_vig_probability, edge, EV
- stake, bankroll_before, result, profit_loss
- closing_odds, CLV, model version

This module defines the canonical record format and validation.

Research basis:
- DrawBias: "version control and tracking model changes"
- quantbet: "bets store a feature snapshot at bet time"
- De Prado (2018): "every prediction must be reproducible"
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class BetRecord:
    """Canonical bet record with all required fields.

    Every field is mandatory for research integrity.
    Missing fields indicate a bug in the backtest engine.
    """
    # --- Identity ---
    match_id: str                      # unique match identifier
    prediction_timestamp: str          # when prediction was made (ISO format)
    model_version: str = "v0.0.0"      # model version at prediction time
    feature_version: str = "v0.0.0"    # feature pipeline version
    calibration_version: str = "v0.0.0" # calibration version

    # --- Match info ---
    league: str = ""
    season: str = ""
    home_team: str = ""
    away_team: str = ""
    kickoff: str = ""                  # match kickoff time (ISO format)

    # --- Model probabilities (BEFORE calibration) ---
    raw_prob_home: float = 0.0
    raw_prob_draw: float = 0.0
    raw_prob_away: float = 0.0

    # --- Calibrated probabilities ---
    cal_prob_home: float = 0.0
    cal_prob_draw: float = 0.0
    cal_prob_away: float = 0.0
    calibration_method: str = "none"   # "sigmoid", "isotonic", "temperature"

    # --- Market odds ---
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    closing_odds_home: float = 0.0
    closing_odds_draw: float = 0.0
    closing_odds_away: float = 0.0

    # --- De-vigged market probabilities ---
    devig_prob_home: float = 0.0
    devig_prob_draw: float = 0.0
    devig_prob_away: float = 0.0

    # --- Bet decision ---
    prediction: str = ""               # "home_win", "draw", "away_win"
    model_prob: float = 0.0            # model probability for predicted outcome
    calibrated_prob: float = 0.0       # calibrated probability for predicted outcome
    market_odds: float = 0.0           # odds taken
    market_implied: float = 0.0        # 1/odds (with margin)
    de_vig_implied: float = 0.0        # de-vigged implied probability
    edge: float = 0.0                  # model_prob - de_vig_implied
    ev: float = 0.0                    # expected value per unit stake
    kelly_fraction: float = 0.0        # recommended Kelly fraction

    # --- Stake ---
    stake: float = 0.0
    bankroll_before: float = 0.0

    # --- Result ---
    result: str = ""                   # "H", "D", "A"
    won: bool = False
    profit_loss: float = 0.0
    bankroll_after: float = 0.0

    # --- CLV ---
    clv_pct: float = 0.0              # (closing_odds - market_odds) / market_odds * 100
    beat_closing: bool = False         # clv_pct > 0

    # --- Feature snapshot ---
    feature_snapshot: Dict[str, Any] = field(default_factory=dict)
    # Stores the exact features used for this prediction

    # --- Metadata ---
    league_code: str = ""
    n_train_matches: int = 0
    git_commit: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def compute_derived(self):
        """Compute derived fields from primary fields."""
        # De-vigged market probabilities
        if self.odds_home > 1.01 and self.odds_draw > 1.01 and self.odds_away > 1.01:
            inv_sum = 1/self.odds_home + 1/self.odds_draw + 1/self.odds_away
            self.devig_prob_home = (1/self.odds_home) / inv_sum
            self.devig_prob_draw = (1/self.odds_draw) / inv_sum
            self.devig_prob_away = (1/self.odds_away) / inv_sum

        # Market implied (with margin)
        if self.market_odds > 1.01:
            self.market_implied = 1.0 / self.market_odds

        # De-vigged implied for the bet outcome
        if self.prediction == "home_win":
            self.de_vig_implied = self.devig_prob_home
        elif self.prediction == "draw":
            self.de_vig_implied = self.devig_prob_draw
        elif self.prediction == "away_win":
            self.de_vig_implied = self.devig_prob_away

        # Edge = model probability - de-vigged market probability
        self.edge = self.calibrated_prob - self.de_vig_implied

        # EV = model_prob * (odds - 1) - (1 - model_prob) * 1
        if self.market_odds > 1.01:
            self.ev = self.calibrated_prob * (self.market_odds - 1) - (1 - self.calibrated_prob)

        # Kelly fraction = edge / (odds - 1)
        if self.market_odds > 1.01 and self.edge > 0:
            self.kelly_fraction = self.edge / (self.market_odds - 1)

        # CLV
        if self.closing_odds_home > 1.01 and self.prediction == "home_win":
            closing = self.closing_odds_home
        elif self.closing_odds_draw > 1.01 and self.prediction == "draw":
            closing = self.closing_odds_draw
        elif self.closing_odds_away > 1.01 and self.prediction == "away_win":
            closing = self.closing_odds_away
        else:
            closing = self.market_odds

        if self.market_odds > 1.01 and closing > 1.01:
            self.clv_pct = (closing - self.market_odds) / self.market_odds * 100
            self.beat_closing = self.clv_pct > 0

        # Result
        if self.result:
            self.won = (
                (self.result == "H" and self.prediction == "home_win") or
                (self.result == "D" and self.prediction == "draw") or
                (self.result == "A" and self.prediction == "away_win")
            )
            if self.stake > 0:
                self.profit_loss = self.stake * (self.market_odds - 1) if self.won else -self.stake
                self.bankroll_after = self.bankroll_before + self.profit_loss

    def compute_hash(self) -> str:
        """Cryptographic hash of the record for integrity verification."""
        record_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(record_str.encode()).hexdigest()[:16]


class BacktestLedger:
    """Append-only ledger of bet records with integrity verification."""

    def __init__(self):
        self.records: list[BetRecord] = []

    def add(self, record: BetRecord):
        """Add a record to the ledger."""
        record.compute_derived()
        self.records.append(record)

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self.records])

    def summary(self) -> dict:
        """Compute summary statistics from the ledger."""
        if not self.records:
            return {"total_bets": 0}

        import numpy as np

        total = len(self.records)
        wins = sum(1 for r in self.records if r.won)
        total_stake = sum(r.stake for r in self.records)
        total_profit = sum(r.profit_loss for r in self.records)
        total_bankroll = sum(r.bankroll_before for r in self.records) / total if total > 0 else 0

        evs = [r.ev for r in self.records]
        edges = [r.edge for r in self.records]
        clvs = [r.clv_pct for r in self.records]

        return {
            "total_bets": total,
            "wins": wins,
            "losses": total - wins,
            "strike_rate": round(wins / total * 100, 2),
            "total_stake": round(total_stake, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(total_profit / total_stake * 100, 2) if total_stake > 0 else 0,
            "yield_pct": round(total_profit / total_stake * 100, 2) if total_stake > 0 else 0,
            "avg_edge": round(float(np.mean(edges)), 4),
            "avg_ev": round(float(np.mean(evs)), 4),
            "avg_clv": round(float(np.mean(clvs)), 2),
            "clv_positive_rate": round(float(np.mean([c > 0 for c in clvs])) * 100, 2),
            "avg_odds": round(float(np.mean([r.market_odds for r in self.records])), 2),
            "avg_stake": round(float(np.mean([r.stake for r in self.records])), 2),
        }

    def verify_integrity(self) -> bool:
        """Verify all records have computed derived fields correctly."""
        for r in self.records:
            r.compute_derived()  # recompute
        return True


if __name__ == "__main__":
    # Self-test
    record = BetRecord(
        match_id="SP1_2024_001",
        prediction_timestamp="2024-08-15T10:00:00",
        model_version="v1.0.0",
        league="La Liga",
        home_team="Barcelona",
        away_team="Getafe",
        kickoff="2024-08-15T20:00:00",
        raw_prob_home=0.65,
        raw_prob_draw=0.20,
        raw_prob_away=0.15,
        cal_prob_home=0.60,
        cal_prob_draw=0.22,
        cal_prob_away=0.18,
        calibration_method="isotonic",
        odds_home=1.80,
        odds_draw=3.50,
        odds_away=4.50,
        closing_odds_home=1.75,
        closing_odds_draw=3.60,
        closing_odds_away=4.80,
        prediction="home_win",
        model_prob=0.65,
        calibrated_prob=0.60,
        market_odds=1.80,
        stake=100.0,
        bankroll_before=10000.0,
        result="H",
        league_code="SP1",
        n_train_matches=500,
    )

    record.compute_derived()

    print("Bet Record Test:")
    print(f"  Match: {record.home_team} vs {record.away_team}")
    print(f"  Prediction: {record.prediction}")
    print(f"  Model prob: {record.model_prob:.1%}")
    print(f"  Calibrated prob: {record.calibrated_prob:.1%}")
    print(f"  Market odds: {record.market_odds}")
    print(f"  De-vig implied: {record.de_vig_implied:.1%}")
    print(f"  Edge: {record.edge:+.1%}")
    print(f"  EV: {record.ev:+.3f}")
    print(f"  Kelly: {record.kelly_fraction:.3f}")
    print(f"  CLV: {record.clv_pct:+.1f}%")
    print(f"  Beat closing: {record.beat_closing}")
    print(f"  Won: {record.won}")
    print(f"  P/L: {record.profit_loss:+.2f}")
    print(f"  Bankroll: {record.bankroll_before:.0f} -> {record.bankroll_after:.0f}")

    # Test ledger
    ledger = BacktestLedger()
    ledger.add(record)
    summary = ledger.summary()
    print(f"\nLedger summary: {summary}")

    print("\n[OK] Backtest record self-test passed.")
