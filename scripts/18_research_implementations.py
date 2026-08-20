#!/usr/bin/env python3
"""
Research Implementations — test the most promising research findings.

Based on literature review, the highest-ROI approaches are:
1. Favourite-longshot bias exploitation (bet on favourites, avoid longshots)
2. Portfolio betting (diversify across markets)
3. Custom objective function (penalize correlation with market)
4. Closing line value filtering (only bet when CLV is positive)
5. Multi-book consensus (bet when multiple books agree)

Usage:
    python scripts/18_research_implementations.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pipeline
from models.poisson_elo_model import PoissonEloModel
from models.ml_layer import MLFootballPredictor


def test_favourite_longshot_bias(df, train_pct=0.65):
    """Test 1: Favourite-longshot bias exploitation.
    
    Research: Snowberg & Levitt (2007), Cain, Law & Peel (2000)
    Bookmakers overprice longshots and underprice favourites.
    Strategy: Only bet on favourites (odds < 2.0), avoid longshots (odds > 4.0).
    """
    print("\n" + "="*70)
    print("TEST 1: FAVOURITE-LONGBIAS EXPLOITATION")
    print("="*70)
    
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n*train_pct)]
    test_df = df.iloc[int(n*0.8):]
    
    poisson = PoissonEloModel(use_dixon_coles=False)
    poisson.train(train_df)
    ml = MLFootballPredictor()
    ml.train(poisson.training_features, verbose=False)
    
    results = {}
    
    for strategy_name, odds_min, odds_max, prob_min in [
        ("Normal (all bets)", 1.6, 100, 0.40),
        ("Favourites only (odds<2.0)", 1.6, 2.0, 0.40),
        ("Mid-range (2.0-3.0)", 2.0, 3.0, 0.40),
        ("Avoid longshots (odds>4.0)", 1.6, 4.0, 0.40),
        ("Strong favourites (odds<1.8, prob>50%)", 1.6, 1.8, 0.50),
    ]:
        bankroll = 1000.0
        bets = 0
        wins = 0
        profits = []
        
        for _, row in test_df.iterrows():
            probs = pipeline.ensemble_probs(poisson, ml, row['home_team'], row['away_team'])
            bookie = {'home_win': row['odds_home_b365'], 'draw': row['odds_draw_b365'],
                      'away_win': row['odds_away_b365']}
            edges = poisson.calculate_edge(probs, bookie, threshold=0.03)
            best = edges.get('best_value')
            if not best or edges.get('max_edge', 0) < 0.03:
                continue
            odds = bookie[best]
            if odds < odds_min or odds > odds_max:
                continue
            if probs[best] < prob_min:
                continue
            edge = edges[best]
            stake = bankroll * pipeline._fractional_kelly(edge, odds)
            if stake < 5:
                continue
            win = pipeline.RESULT_MAP.get(row['result']) == best
            profit = stake * (odds - 1) if win else -stake
            bankroll += profit
            profits.append(profit)
            bets += 1
            if win:
                wins += 1
        
        roi = (bankroll - 1000) / 1000 * 100
        wr = wins / bets * 100 if bets > 0 else 0
        avg_profit = np.mean(profits) if profits else 0
        results[strategy_name] = {
            'roi': roi, 'bets': bets, 'win_rate': wr,
            'final': bankroll, 'avg_profit': avg_profit
        }
        print(f"  {strategy_name:<35} ROI={roi:+6.1f}%  bets={bets:3d}  WR={wr:5.1f}%  final=${bankroll:,.0f}")
    
    return results


def test_consensus_betting(df, train_pct=0.65):
    """Test 2: Multi-book consensus betting.
    
    Research: Market consensus is a strong predictor.
    Strategy: Only bet when model AND bookmaker agree on the outcome.
    """
    print("\n" + "="*70)
    print("TEST 2: CONSENSUS BETTING (Model + Market agree)")
    print("="*70)
    
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n*train_pct)]
    test_df = df.iloc[int(n*0.8):]
    
    poisson = PoissonEloModel(use_dixon_coles=False)
    poisson.train(train_df)
    ml = MLFootballPredictor()
    ml.train(poisson.training_features, verbose=False)
    
    results = {}
    
    for strategy_name, require_consensus in [
        ("Normal (no consensus)", False),
        ("Consensus required", True),
        ("Strong consensus (both >55%)", True),
    ]:
        bankroll = 1000.0
        bets = 0
        wins = 0
        
        for _, row in test_df.iterrows():
            probs = pipeline.ensemble_probs(poisson, ml, row['home_team'], row['away_team'])
            bookie = {'home_win': row['odds_home_b365'], 'draw': row['odds_draw_b365'],
                      'away_win': row['odds_away_b365']}
            edges = poisson.calculate_edge(probs, bookie, threshold=0.03)
            best = edges.get('best_value')
            if not best or edges.get('max_edge', 0) < 0.03:
                continue
            odds = bookie[best]
            if odds < 1.6 or probs[best] < 0.40:
                continue
            
            # Consensus check
            if require_consensus:
                # Model says this outcome
                model_best = max(probs, key=probs.get)
                # Market says this outcome (highest implied probability)
                from models.calibration import implied_probs
                market_probs = implied_probs(bookie)
                market_best = max(market_probs, key=market_probs.get)
                
                if strategy_name == "Strong consensus (both >55%)":
                    if probs[best] < 0.55 or market_probs.get(best, 0) < 0.55:
                        continue
                else:
                    if model_best != best or market_best != best:
                        continue
            
            edge = edges[best]
            stake = bankroll * pipeline._fractional_kelly(edge, odds)
            if stake < 5:
                continue
            win = pipeline.RESULT_MAP.get(row['result']) == best
            profit = stake * (odds - 1) if win else -stake
            bankroll += profit
            bets += 1
            if win:
                wins += 1
        
        roi = (bankroll - 1000) / 1000 * 100
        wr = wins / bets * 100 if bets > 0 else 0
        results[strategy_name] = {'roi': roi, 'bets': bets, 'win_rate': wr, 'final': bankroll}
        print(f"  {strategy_name:<35} ROI={roi:+6.1f}%  bets={bets:3d}  WR={wr:5.1f}%  final=${bankroll:,.0f}")
    
    return results


def test_edge_sizing(df, train_pct=0.65):
    """Test 3: Edge-proportional staking.
    
    Research: Kelly criterion is optimal but estimation error hurts.
    Strategy: Stake proportional to edge size (bigger edge = bigger bet).
    """
    print("\n" + "="*70)
    print("TEST 3: EDGE-PROPORTIONAL STAKING")
    print("="*70)
    
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n*train_pct)]
    test_df = df.iloc[int(n*0.8):]
    
    poisson = PoissonEloModel(use_dixon_coles=False)
    poisson.train(train_df)
    ml = MLFootballPredictor()
    ml.train(poisson.training_features, verbose=False)
    
    results = {}
    
    for strategy_name, kelly_frac, max_stake in [
        ("Quarter Kelly, 8% cap", 0.25, 0.08),
        ("Half Kelly, 8% cap", 0.50, 0.08),
        ("Quarter Kelly, 15% cap", 0.25, 0.15),
        ("Flat 2% per bet", 0.0, 0.02),  # special case
    ]:
        bankroll = 1000.0
        bets = 0
        wins = 0
        
        for _, row in test_df.iterrows():
            probs = pipeline.ensemble_probs(poisson, ml, row['home_team'], row['away_team'])
            bookie = {'home_win': row['odds_home_b365'], 'draw': row['odds_draw_b365'],
                      'away_win': row['odds_away_b365']}
            edges = poisson.calculate_edge(probs, bookie, threshold=0.03)
            best = edges.get('best_value')
            if not best or edges.get('max_edge', 0) < 0.03:
                continue
            odds = bookie[best]
            if odds < 1.6 or probs[best] < 0.40:
                continue
            edge = edges[best]
            
            if kelly_frac == 0.0:
                # Flat staking
                stake = bankroll * max_stake
            else:
                stake = bankroll * pipeline._fractional_kelly(edge, odds, fraction=kelly_frac)
                stake = min(stake, bankroll * max_stake)
            
            if stake < 5:
                continue
            win = pipeline.RESULT_MAP.get(row['result']) == best
            profit = stake * (odds - 1) if win else -stake
            bankroll += profit
            bets += 1
            if win:
                wins += 1
        
        roi = (bankroll - 1000) / 1000 * 100
        wr = wins / bets * 100 if bets > 0 else 0
        results[strategy_name] = {'roi': roi, 'bets': bets, 'win_rate': wr, 'final': bankroll}
        print(f"  {strategy_name:<35} ROI={roi:+6.1f}%  bets={bets:3d}  WR={wr:5.1f}%  final=${bankroll:,.0f}")
    
    return results


def test_market_edge_filter(df, train_pct=0.65):
    """Test 4: Market edge filtering.
    
    Research: CLV is the strongest predictor of profitability.
    Strategy: Only bet when the model disagrees significantly with the market.
    """
    print("\n" + "="*70)
    print("TEST 4: MARKET EDGE FILTERING")
    print("="*70)
    
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n*train_pct)]
    test_df = df.iloc[int(n*0.8):]
    
    poisson = PoissonEloModel(use_dixon_coles=False)
    poisson.train(train_df)
    ml = MLFootballPredictor()
    ml.train(poisson.training_features, verbose=False)
    
    from models.calibration import implied_probs
    
    results = {}
    
    for strategy_name, min_prob_diff, min_edge in [
        ("Normal (3% edge)", 0.0, 0.03),
        ("Model-market diff >5%", 0.05, 0.03),
        ("Model-market diff >10%", 0.10, 0.03),
        ("Edge >5% (strong signal)", 0.0, 0.05),
        ("Edge >8% (very strong)", 0.0, 0.08),
    ]:
        bankroll = 1000.0
        bets = 0
        wins = 0
        
        for _, row in test_df.iterrows():
            probs = pipeline.ensemble_probs(poisson, ml, row['home_team'], row['away_team'])
            bookie = {'home_win': row['odds_home_b365'], 'draw': row['odds_draw_b365'],
                      'away_win': row['odds_away_b365']}
            edges = poisson.calculate_edge(probs, bookie, threshold=min_edge)
            best = edges.get('best_value')
            if not best or edges.get('max_edge', 0) < min_edge:
                continue
            odds = bookie[best]
            if odds < 1.6 or probs[best] < 0.40:
                continue
            
            # Market edge filter
            market_probs = implied_probs(bookie)
            model_prob = probs[best]
            market_prob = market_probs.get(best, 0.33)
            prob_diff = abs(model_prob - market_prob)
            
            if prob_diff < min_prob_diff:
                continue
            
            edge = edges[best]
            stake = bankroll * pipeline._fractional_kelly(edge, odds)
            if stake < 5:
                continue
            win = pipeline.RESULT_MAP.get(row['result']) == best
            profit = stake * (odds - 1) if win else -stake
            bankroll += profit
            bets += 1
            if win:
                wins += 1
        
        roi = (bankroll - 1000) / 1000 * 100
        wr = wins / bets * 100 if bets > 0 else 0
        results[strategy_name] = {'roi': roi, 'bets': bets, 'win_rate': wr, 'final': bankroll}
        print(f"  {strategy_name:<35} ROI={roi:+6.1f}%  bets={bets:3d}  WR={wr:5.1f}%  final=${bankroll:,.0f}")
    
    return results


def main():
    print("="*70)
    print("RESEARCH IMPLEMENTATIONS — TESTING HIGH-ROI STRATEGIES")
    print("="*70)
    
    df = pipeline.generate_match_data(1200, seed=42)
    print(f"Generated {len(df)} matches")
    
    all_results = {}
    
    # Run all tests
    all_results['favourite_longshot'] = test_favourite_longshot_bias(df)
    all_results['consensus'] = test_consensus_betting(df)
    all_results['edge_sizing'] = test_edge_sizing(df)
    all_results['market_edge'] = test_market_edge_filter(df)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: BEST STRATEGIES BY CATEGORY")
    print("="*70)
    
    for category, results in all_results.items():
        best = max(results.items(), key=lambda x: x[1]['roi'])
        print(f"\n  {category.upper()}:")
        print(f"    Best: {best[0]}")
        print(f"    ROI: {best[1]['roi']:+.1f}%, Bets: {best[1]['bets']}, WR: {best[1]['win_rate']:.1f}%")
    
    # Overall best
    all_strats = []
    for category, results in all_results.items():
        for name, metrics in results.items():
            all_strats.append((category, name, metrics))
    
    overall_best = max(all_strats, key=lambda x: x[2]['roi'])
    print(f"\n{'='*70}")
    print(f"OVERALL BEST STRATEGY:")
    print(f"  Category: {overall_best[0]}")
    print(f"  Strategy: {overall_best[1]}")
    print(f"  ROI:      {overall_best[2]['roi']:+.1f}%")
    print(f"  Bets:     {overall_best[2]['bets']}")
    print(f"  Win Rate: {overall_best[2]['win_rate']:.1f}%")
    print(f"  Final:    ${overall_best[2]['final']:,.0f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
