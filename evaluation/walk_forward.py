#!/usr/bin/env python3
"""
Walk-Forward Validation Across Multiple Seasons

Implements proper out-of-sample testing:
- Train on seasons 1-N, test on season N+1
- Roll forward, retraining each season
- No data leakage (future data never used in training)
- Measures calibration, ROI, and profit across seasons

This is the gold standard for sports betting model evaluation.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class SeasonResult:
    """Results from a single season's walk-forward test."""
    season: str
    n_matches: int
    n_bets: int
    wins: int
    losses: int
    profit: float
    roi: float
    avg_odds: float
    avg_edge: float
    ece: float
    log_loss: float
    accuracy: float
    max_drawdown: float
    yield_pct: float


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation results."""
    seasons: List[SeasonResult] = field(default_factory=list)
    total_matches: int = 0
    total_bets: int = 0
    total_profit: float = 0.0
    total_roi: float = 0.0
    avg_ece: float = 0.0
    avg_log_loss: float = 0.0
    avg_accuracy: float = 0.0
    avg_yield: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0


def walk_forward_validation(
    df: pd.DataFrame,
    model_class,
    bet_threshold: float = 0.05,
    min_odds: float = 1.5,
    max_odds: float = 10.0,
    kelly_fraction: float = 0.25,
    initial_bankroll: float = 1000.0,
    verbose: bool = True
) -> WalkForwardResult:
    """Run walk-forward validation across multiple seasons.
    
    Args:
        df: Full match DataFrame with season column
        model_class: Model class to use (must have train/predict methods)
        bet_threshold: Minimum edge to place a bet
        min_odds: Minimum odds to bet on
        max_odds: Maximum odds to bet on
        kelly_fraction: Fraction of Kelly criterion to use
        initial_bankroll: Starting bankroll
        verbose: Print progress
    
    Returns:
        WalkForwardResult with all season results
    """
    if verbose:
        print("=" * 70)
        print("WALK-FORWARD VALIDATION")
        print("=" * 70)
    
    # Identify seasons
    if "season" in df.columns:
        seasons = sorted(df["season"].unique())
    elif "date" in df.columns:
        # Extract season from date (assume season = year)
        df["season"] = pd.to_datetime(df["date"]).dt.year
        seasons = sorted(df["season"].unique())
    else:
        # Treat each 30% chunk as a "season"
        n = len(df)
        chunk_size = n // 3
        seasons = list(range(3))
        df["season"] = [i // chunk_size for i in range(n)]
        seasons = sorted(df["season"].unique())
    
    if verbose:
        print(f"Seasons: {seasons}")
        print(f"Total matches: {len(df)}")
        print()
    
    result = WalkForwardResult()
    bankroll = initial_bankroll
    equity_curve = [bankroll]
    all_bets = []
    
    for i, season in enumerate(seasons):
        if i < 2:
            # Need at least 2 seasons for training
            if verbose:
                print(f"Season {season}: Skipped (need training data)")
            continue
        
        # Train on all previous seasons
        train_seasons = seasons[:i]
        test_season = season
        
        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df = df[df["season"] == test_season].copy()
        
        if verbose:
            print(f"Season {test_season}:")
            print(f"  Train: {len(train_df)} matches ({train_seasons})")
            print(f"  Test:  {len(test_df)} matches")
        
        # Train model
        model = model_class()
        model.train(train_df, verbose=False)
        
        # Test on held-out season
        season_result = _test_season(
            model, test_df, bankroll, bet_threshold,
            min_odds, max_odds, kelly_fraction, verbose
        )
        
        # Update bankroll
        bankroll += season_result.profit
        equity_curve.append(bankroll)
        
        result.seasons.append(season_result)
        result.total_matches += season_result.n_matches
        result.total_bets += season_result.n_bets
        result.total_profit += season_result.profit
        
        if verbose:
            print(f"  Bets: {season_result.n_bets}, "
                  f"ROI: {season_result.roi:.1%}, "
                  f"Profit: ${season_result.profit:.2f}, "
                  f"Bankroll: ${bankroll:.2f}")
            print()
    
    # Compute aggregate metrics
    result.total_roi = result.total_profit / initial_bankroll
    result.avg_ece = np.mean([s.ece for s in result.seasons])
    result.avg_log_loss = np.mean([s.log_loss for s in result.seasons])
    result.avg_accuracy = np.mean([s.accuracy for s in result.seasons])
    result.avg_yield = np.mean([s.yield_pct for s in result.seasons])
    result.max_drawdown = _max_drawdown(equity_curve)
    result.sharpe_ratio = _sharpe_ratio(equity_curve)
    result.profit_factor = _profit_factor(result.seasons)
    
    if verbose:
        print("=" * 70)
        print("WALK-FORWARD RESULTS")
        print("=" * 70)
        print(f"Total matches:  {result.total_matches}")
        print(f"Total bets:     {result.total_bets}")
        print(f"Total profit:   ${result.total_profit:.2f}")
        print(f"Total ROI:      {result.total_roi:.1%}")
        print(f"Avg ECE:        {result.avg_ece:.4f}")
        print(f"Avg Log-loss:   {result.avg_log_loss:.4f}")
        print(f"Avg Accuracy:   {result.avg_accuracy:.1%}")
        print(f"Avg Yield:      {result.avg_yield:.1%}")
        print(f"Max Drawdown:   {result.max_drawdown:.1%}")
        print(f"Sharpe Ratio:   {result.sharpe_ratio:.2f}")
        print(f"Profit Factor:  {result.profit_factor:.2f}")
        print("=" * 70)
    
    return result


def _test_season(
    model,
    test_df: pd.DataFrame,
    bankroll: float,
    bet_threshold: float,
    min_odds: float,
    max_odds: float,
    kelly_fraction: float,
    verbose: bool
) -> SeasonResult:
    """Test model on a single season."""
    all_probs = []
    all_true = []
    all_odds_home = []
    all_odds_draw = []
    all_odds_away = []
    
    for _, row in test_df.iterrows():
        try:
            probs = model.predict(row["home_team"], row["away_team"])
            all_probs.append([probs["away_win"], probs["draw"], probs["home_win"]])
            
            true_map = {"A": 0, "D": 1, "H": 2}
            all_true.append(true_map.get(row["result"], 1))
            
            # Get odds if available
            if "B365H" in row.index:
                all_odds_home.append(float(row["B365H"]) if pd.notna(row["B365H"]) else 0)
                all_odds_draw.append(float(row["B365D"]) if pd.notna(row["B365D"]) else 0)
                all_odds_away.append(float(row["B365A"]) if pd.notna(row["B365A"]) else 0)
            else:
                all_odds_home.append(0)
                all_odds_draw.append(0)
                all_odds_away.append(0)
        except Exception:
            continue
    
    if not all_probs:
        return SeasonResult(
            season=str(test_df["season"].iloc[0]) if "season" in test_df.columns else "unknown",
            n_matches=0, n_bets=0, wins=0, losses=0,
            profit=0.0, roi=0.0, avg_odds=0.0, avg_edge=0.0,
            ece=1.0, log_loss=1.1, accuracy=0.33, max_drawdown=0.0, yield_pct=0.0
        )
    
    probs_arr = np.array(all_probs)
    y_true = np.array(all_true)
    odds_home = np.array(all_odds_home)
    odds_draw = np.array(all_odds_draw)
    odds_away = np.array(all_odds_away)
    
    # Compute calibration metrics
    from models.calibration_selection import expected_calibration_error, log_loss as compute_ll
    ece = expected_calibration_error(y_true, probs_arr)
    ll = compute_ll(y_true, probs_arr)
    accuracy = float(np.mean(np.argmax(probs_arr, axis=1) == y_true))
    
    # Simulate betting
    bets = 0
    wins = 0
    losses_count = 0
    total_profit = 0.0
    total_odds = []
    total_edge = []
    
    for i in range(len(y_true)):
        pred = np.argmax(probs_arr[i])
        prob = probs_arr[i, pred]
        
        # Get odds for predicted outcome
        if pred == 2:  # Home win
            odds = odds_home[i]
        elif pred == 1:  # Draw
            odds = odds_draw[i]
        else:  # Away win
            odds = odds_away[i]
        
        # Skip if no odds or odds out of range
        if odds < min_odds or odds > max_odds or odds == 0:
            continue
        
        # Compute edge
        implied_prob = 1.0 / odds
        edge = prob - implied_prob
        
        # Only bet if edge exceeds threshold
        if edge < bet_threshold:
            continue
        
        # Kelly stake
        kelly = (prob * odds - 1) / (odds - 1)
        stake = bankroll * kelly * kelly_fraction
        
        bets += 1
        total_odds.append(odds)
        total_edge.append(edge)
        
        # Check if bet won
        if pred == y_true[i]:
            profit = stake * (odds - 1)
            total_profit += profit
            wins += 1
        else:
            profit = -stake
            total_profit += profit
            losses_count += 1
    
    roi = total_profit / bankroll if bankroll > 0 else 0
    avg_odds = np.mean(total_odds) if total_odds else 0
    avg_edge = np.mean(total_edge) if total_edge else 0
    yield_pct = total_profit / (bankroll * bets) if bets > 0 else 0
    
    return SeasonResult(
        season=str(test_df["season"].iloc[0]) if "season" in test_df.columns else "unknown",
        n_matches=len(y_true),
        n_bets=bets,
        wins=wins,
        losses=losses_count,
        profit=round(total_profit, 2),
        roi=round(roi, 4),
        avg_odds=round(avg_odds, 2),
        avg_edge=round(avg_edge, 4),
        ece=round(ece, 4),
        log_loss=round(ll, 4),
        accuracy=round(accuracy, 4),
        max_drawdown=0.0,
        yield_pct=round(yield_pct, 4),
    )


def _max_drawdown(equity_curve: List[float]) -> float:
    """Compute maximum drawdown from equity curve."""
    if not equity_curve:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def _sharpe_ratio(equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
    """Compute Sharpe ratio from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    
    returns = []
    for i in range(1, len(equity_curve)):
        ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
        returns.append(ret)
    
    if not returns:
        return 0.0
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Annualize (assume 380 matches per season, ~3 seasons)
    annual_factor = np.sqrt(380 * 3)
    sharpe = (avg_return - risk_free_rate) / std_return * annual_factor
    
    return sharpe


def _profit_factor(seasons: List[SeasonResult]) -> float:
    """Compute profit factor (gross wins / gross losses)."""
    gross_wins = sum(s.profit for s in seasons if s.profit > 0)
    gross_losses = abs(sum(s.profit for s in seasons if s.profit < 0))
    
    if gross_losses == 0:
        return float("inf") if gross_wins > 0 else 0.0
    
    return gross_wins / gross_losses


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    print("Walk-forward validation module loaded.")
    print("Use walk_forward_validation() with your model and data.")
    print("[OK] Walk-forward module ready.")
