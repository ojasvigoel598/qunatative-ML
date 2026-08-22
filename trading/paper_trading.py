#!/usr/bin/env python3
"""
Paper Trading System

Track live predictions without risking real money.
Records:
- Model predictions at bet time
- Odds at bet time
- Actual outcome
- Profit/loss
- Calibration over time

This is essential for validating that your model works in practice,
not just in backtests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np


TRADES_FILE = Path(__file__).parent.parent / "data" / "paper_trades.json"


@dataclass
class PaperTrade:
    """A single paper trade."""
    trade_id: str
    timestamp: str
    
    # Match info
    match_id: str
    home_team: str
    away_team: str
    date: str
    league: str
    
    # Prediction
    model_home_prob: float
    model_draw_prob: float
    model_away_prob: float
    predicted_outcome: str  # "H", "D", "A"
    confidence: float
    
    # Odds
    odds_home: float
    odds_draw: float
    odds_away: float
    odds_at_bet: float
    
    # Edge
    edge: float
    kelly_stake: float
    
    # Outcome (filled after match)
    actual_outcome: Optional[str] = None
    won: Optional[bool] = None
    profit: Optional[float] = None
    
    # CLV (filled when closing odds available)
    closing_odds_home: Optional[float] = None
    closing_odds_draw: Optional[float] = None
    closing_odds_away: Optional[float] = None
    clv: Optional[float] = None


@dataclass
class PaperTradingState:
    """State of the paper trading system."""
    trades: List[PaperTrade] = field(default_factory=list)
    bankroll: float = 1000.0
    initial_bankroll: float = 1000.0
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def resolved_trades(self) -> int:
        return sum(1 for t in self.trades if t.actual_outcome is not None)
    
    @property
    def total_profit(self) -> float:
        return sum(t.profit for t in self.trades if t.profit is not None)
    
    @property
    def roi(self) -> float:
        if self.initial_bankroll == 0:
            return 0.0
        return self.total_profit / self.initial_bankroll
    
    @property
    def win_rate(self) -> float:
        resolved = [t for t in self.trades if t.won is not None]
        if not resolved:
            return 0.0
        return sum(1 for t in resolved if t.won) / len(resolved)
    
    @property
    def avg_edge(self) -> float:
        edges = [t.edge for t in self.trades]
        return np.mean(edges) if edges else 0.0
    
    @property
    def avg_confidence(self) -> float:
        confs = [t.confidence for t in self.trades]
        return np.mean(confs) if confs else 0.0
    
    @property
    def calibration_error(self) -> float:
        """Compute calibration error from resolved trades."""
        resolved = [t for t in self.trades if t.actual_outcome is not None]
        if not resolved:
            return 1.0
        
        # Group by predicted probability bins
        bins = {}
        for trade in resolved:
            prob = trade.confidence
            bin_idx = int(prob * 10) / 10  # Round to nearest 0.1
            if bin_idx not in bins:
                bins[bin_idx] = {"predicted": [], "actual": []}
            bins[bin_idx]["predicted"].append(prob)
            bins[bin_idx]["actual"].append(1.0 if trade.won else 0.0)
        
        # Compute ECE
        ece = 0.0
        for bin_idx, data in bins.items():
            if data["predicted"]:
                avg_predicted = np.mean(data["predicted"])
                avg_actual = np.mean(data["actual"])
                weight = len(data["predicted"]) / len(resolved)
                ece += weight * abs(avg_predicted - avg_actual)
        
        return ece
    
    def summary(self) -> str:
        """Generate summary report."""
        return f"""
{'='*60}
PAPER TRADING SUMMARY
{'='*60}

Total Trades:     {self.total_trades}
Resolved Trades:  {self.resolved_trades}
Pending Trades:   {self.total_trades - self.resolved_trades}

Performance:
  Total Profit:   ${self.total_profit:.2f}
  ROI:            {self.roi:.1%}
  Win Rate:       {self.win_rate:.1%}
  Avg Edge:       {self.avg_edge:.4f}
  Avg Confidence: {self.avg_confidence:.4f}
  
Calibration:
  ECE:            {self.calibration_error:.4f}
  
Bankroll:
  Initial:        ${self.initial_bankroll:.2f}
  Current:        ${self.bankroll:.2f}
{'='*60}
"""


class PaperTradingSystem:
    """Paper trading system for tracking live predictions."""
    
    def __init__(self, bankroll: float = 1000.0):
        self.state = PaperTradingState(
            bankroll=bankroll,
            initial_bankroll=bankroll
        )
        self.trade_counter = 0
        self._load()
    
    def _load(self):
        """Load trades from file."""
        if TRADES_FILE.exists():
            try:
                with open(TRADES_FILE) as f:
                    data = json.load(f)
                self.state.bankroll = data.get("bankroll", self.state.bankroll)
                self.state.initial_bankroll = data.get("initial_bankroll", self.state.initial_bankroll)
                for trade_data in data.get("trades", []):
                    self.state.trades.append(PaperTrade(**trade_data))
                self.trade_counter = len(self.state.trades)
            except Exception:
                pass
    
    def _save(self):
        """Save trades to file."""
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bankroll": self.state.bankroll,
            "initial_bankroll": self.state.initial_bankroll,
            "trades": [asdict(t) for t in self.state.trades]
        }
        with open(TRADES_FILE, "w") as f:
            json.dump(data, f, indent=2)
    
    def place_trade(
        self,
        home_team: str,
        away_team: str,
        date: str,
        league: str,
        model_probs: Dict[str, float],
        odds: Dict[str, float],
        edge: float,
        kelly_stake: float,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.25
    ) -> Optional[PaperTrade]:
        """Place a paper trade.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            date: Match date
            league: League name
            model_probs: {"home": 0.5, "draw": 0.25, "away": 0.25}
            odds: {"home": 2.0, "draw": 3.5, "away": 3.8}
            edge: Calculated edge
            kelly_stake: Kelly stake size
            min_edge: Minimum edge to place trade
            kelly_fraction: Fraction of Kelly to use
        
        Returns:
            PaperTrade if trade placed, None if skipped
        """
        if edge < min_edge:
            return None
        
        # Determine predicted outcome
        probs = [model_probs["home"], model_probs["draw"], model_probs["away"]]
        outcomes = ["H", "D", "A"]
        pred_idx = np.argmax(probs)
        predicted_outcome = outcomes[pred_idx]
        confidence = probs[pred_idx]
        
        # Get odds for predicted outcome
        if predicted_outcome == "H":
            odds_at_bet = odds["home"]
        elif predicted_outcome == "D":
            odds_at_bet = odds["draw"]
        else:
            odds_at_bet = odds["away"]
        
        # Create trade
        self.trade_counter += 1
        trade = PaperTrade(
            trade_id=f"PT-{self.trade_counter:06d}",
            timestamp=datetime.now().isoformat(),
            match_id=f"{home_team}_vs_{away_team}_{date}",
            home_team=home_team,
            away_team=away_team,
            date=date,
            league=league,
            model_home_prob=model_probs["home"],
            model_draw_prob=model_probs["draw"],
            model_away_prob=model_probs["away"],
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            odds_home=odds["home"],
            odds_draw=odds["draw"],
            odds_away=odds["away"],
            odds_at_bet=odds_at_bet,
            edge=edge,
            kelly_stake=kelly_stake * kelly_fraction,
        )
        
        self.state.trades.append(trade)
        self._save()
        
        return trade
    
    def resolve_trade(
        self,
        trade_id: str,
        actual_outcome: str,
        closing_odds: Optional[Dict[str, float]] = None
    ) -> bool:
        """Resolve a trade with actual outcome.
        
        Args:
            trade_id: Trade ID to resolve
            actual_outcome: "H", "D", or "A"
            closing_odds: Optional closing odds for CLV
        
        Returns:
            True if resolved, False if trade not found
        """
        for trade in self.state.trades:
            if trade.trade_id == trade_id:
                trade.actual_outcome = actual_outcome
                trade.won = (trade.predicted_outcome == actual_outcome)
                
                # Compute profit
                stake = trade.kelly_stake
                if trade.won:
                    trade.profit = stake * (trade.odds_at_bet - 1)
                else:
                    trade.profit = -stake
                
                self.state.bankroll += trade.profit
                
                # Compute CLV if closing odds available
                if closing_odds:
                    trade.closing_odds_home = closing_odds.get("home")
                    trade.closing_odds_draw = closing_odds.get("draw")
                    trade.closing_odds_away = closing_odds.get("away")
                    
                    # CLV for the outcome we bet on
                    model_prob = getattr(trade, f"model_{trade.predicted_outcome.lower()}_prob")
                    closing_odds_for_outcome = closing_odds.get(
                        "home" if trade.predicted_outcome == "H" else
                        "draw" if trade.predicted_outcome == "D" else "away"
                    )
                    if closing_odds_for_outcome and closing_odds_for_outcome > 1:
                        implied_prob = 1 / closing_odds_for_outcome
                        trade.clv = (model_prob - implied_prob) / implied_prob
                
                self._save()
                return True
        
        return False
    
    def get_trades_by_league(self, league: str) -> List[PaperTrade]:
        """Get all trades for a specific league."""
        return [t for t in self.state.trades if t.league == league]
    
    def get_trades_by_team(self, team: str) -> List[PaperTrade]:
        """Get all trades involving a specific team."""
        return [t for t in self.state.trades 
                if t.home_team == team or t.away_team == team]
    
    def get_recent_trades(self, n: int = 10) -> List[PaperTrade]:
        """Get the N most recent trades."""
        return self.state.trades[-n:]
    
    def summary(self) -> str:
        """Get summary report."""
        return self.state.summary()


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    print("Testing paper trading system...")
    
    system = PaperTradingSystem(bankroll=1000.0)
    
    # Place some trades
    trade1 = system.place_trade(
        home_team="Barcelona",
        away_team="Real Madrid",
        date="2024-01-15",
        league="La Liga",
        model_probs={"home": 0.55, "draw": 0.25, "away": 0.20},
        odds={"home": 2.0, "draw": 3.5, "away": 3.8},
        edge=0.05,
        kelly_stake=0.025
    )
    
    trade2 = system.place_trade(
        home_team="Man City",
        away_team="Liverpool",
        date="2024-01-16",
        league="EPL",
        model_probs={"home": 0.45, "draw": 0.30, "away": 0.25},
        odds={"home": 2.2, "draw": 3.3, "away": 3.5},
        edge=0.06,
        kelly_stake=0.03
    )
    
    if trade1:
        print(f"  Trade 1 placed: {trade1.trade_id}")
        system.resolve_trade(trade1.trade_id, "H")  # Home won
    
    if trade2:
        print(f"  Trade 2 placed: {trade2.trade_id}")
        system.resolve_trade(trade2.trade_id, "D")  # Draw
    
    print(f"\n{system.summary()}")
    
    print("[OK] Paper trading system complete.")
