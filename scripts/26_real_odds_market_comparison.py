#!/usr/bin/env python3
"""
Real Odds Market Comparison — Test on Completely New Leagues.

Uses trained model on known leagues (La Liga, EPL, Serie A)
and tests on completely unseen leagues (Bundesliga, Ligue 1, Eredivisie).

This tests genuine out-of-sample performance on new competitions.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from data.real_data import load_league, LEAGUES
from models.layered_model import LayeredModel
from models.calibration import expected_calibration_error


def implied_prob(odds):
    """Convert decimal odds to implied probability."""
    if pd.isna(odds) or odds <= 1.0:
        return np.nan
    return 1.0 / odds


def de_vig_probs(home_odds, draw_odds, away_odds):
    """Remove bookmaker margin from odds."""
    p_h = implied_prob(home_odds)
    p_d = implied_prob(draw_odds)
    p_a = implied_prob(away_odds)

    if pd.isna(p_h) or pd.isna(p_d) or pd.isna(p_a):
        return np.nan, np.nan, np.nan

    overround = p_h + p_d + p_a
    if overround <= 0:
        return np.nan, np.nan, np.nan

    return p_h / overround, p_d / overround, p_a / overround


def run_market_comparison():
    """Test model on real odds across multiple leagues."""
    print("=" * 70)
    print("REAL ODDS MARKET COMPARISON")
    print("Train on La Liga, EPL, Serie A | Test on Bundesliga, Ligue 1")
    print("=" * 70)

    # Train on known leagues
    print("\nLoading training data...")
    train_leagues = ["SP1", "E0", "I1"]
    train_dfs = []
    for league in train_leagues:
        try:
            df = load_league(league, seasons=None, offline=True)
            if len(df) > 0:
                train_dfs.append(df)
                print(f"  {league}: {len(df)} matches")
        except Exception as e:
            print(f"  {league}: Failed ({e})")

    if not train_dfs:
        print("No training data!")
        return

    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = train_df.sort_values("date").reset_index(drop=True)
    print(f"  Total training: {len(train_df)} matches")

    # Train model
    print("\nTraining model...")
    t0 = time.perf_counter()
    model = LayeredModel(layers=["bayesian", "ewma"])
    model.train(train_df, verbose=True)
    t_train = (time.perf_counter() - t0) * 1000
    print(f"  Training time: {t_train:.0f}ms")

    # Test on completely new leagues (load directly from CSV)
    test_leagues = ["D1", "F1"]  # Bundesliga, Ligue 1 (NOT in training)
    test_names = {"D1": "Bundesliga", "F1": "Ligue 1"}

    all_results = []

    for league in test_leagues:
        print(f"\n{'=' * 70}")
        print(f"TESTING ON: {test_names[league]} (completely unseen)")
        print(f"{'=' * 70}")

        try:
            # Load directly from CSV since these leagues aren't in LEAGUES dict
            test_df = pd.read_csv(f"data/real/{league}_2425.csv")
            # Rename columns to match schema
            test_df = test_df.rename(columns={
                "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
                "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
                "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
            })
            test_df["date"] = pd.to_datetime(test_df["date"], format="%d/%m/%Y", errors="coerce")
            test_df = test_df.dropna(subset=["home_goals", "away_goals", "result"])
            test_df["result"] = test_df["result"].map({"H": "H", "D": "D", "A": "A"})
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        if len(test_df) == 0:
            print("  No data!")
            continue

        print(f"  Matches: {len(test_df)}")

        # Filter to matches with odds
        test_df = test_df.dropna(subset=["odds_home", "odds_draw", "odds_away"])
        print(f"  With odds: {len(test_df)}")

        # Run predictions
        model_probs = []
        market_probs = []
        true_results = []
        odds_list = []

        for _, row in test_df.iterrows():
            try:
                # Model prediction
                probs = model.predict(row["home_team"], row["away_team"])
                model_probs.append([probs["home_win"], probs["draw"], probs["away_win"]])

                # Market de-vigged probabilities
                m_h, m_d, m_a = de_vig_probs(
                    row["odds_home"], row["odds_draw"], row["odds_away"]
                )
                market_probs.append([m_h, m_d, m_a])

                # True result
                true_map = {"H": 0, "D": 1, "A": 2}
                true_results.append(true_map.get(row["result"], 1))

                odds_list.append({
                    "home": row["odds_home"],
                    "draw": row["odds_draw"],
                    "away": row["odds_away"],
                })
            except Exception:
                continue

        if len(model_probs) < 10:
            print("  Too few predictions!")
            continue

        model_arr = np.array(model_probs)
        market_arr = np.array(market_probs)
        y_true = np.array(true_results)

        # Metrics
        eps = 1e-9
        model_ll = float(-np.mean(np.log(np.clip(model_arr[np.arange(len(y_true)), y_true], eps, 1))))
        market_ll = float(-np.mean(np.log(np.clip(market_arr[np.arange(len(y_true)), y_true], eps, 1))))

        model_brier = float(np.mean(np.sum((model_arr - np.eye(3)[y_true]) ** 2, axis=1)))
        market_brier = float(np.mean(np.sum((market_arr - np.eye(3)[y_true]) ** 2, axis=1)))

        model_acc = float(np.mean(np.argmax(model_arr, axis=1) == y_true))
        market_acc = float(np.mean(np.argmax(market_arr, axis=1) == y_true))

        model_ece = expected_calibration_error(model_arr, y_true)
        market_ece = expected_calibration_error(market_arr, y_true)

        # ROI simulation (flat stake on model's top pick vs market's top pick)
        model_roi = simulate_roi(model_arr, y_true, odds_list, "model")
        market_roi = simulate_roi(market_arr, y_true, odds_list, "market")

        print(f"\n  MODEL vs MARKET on {test_names[league]}:")
        print(f"  {'Metric':<20} {'Model':>10} {'Market':>10} {'Winner':>10}")
        print(f"  {'-' * 50}")
        print(f"  {'Log-loss':<20} {model_ll:>10.4f} {market_ll:>10.4f} {'Model' if model_ll < market_ll else 'Market':>10}")
        print(f"  {'Brier':<20} {model_brier:>10.4f} {market_brier:>10.4f} {'Model' if model_brier < market_brier else 'Market':>10}")
        print(f"  {'Accuracy':<20} {model_acc:>9.1%} {market_acc:>9.1%} {'Model' if model_acc > market_acc else 'Market':>10}")
        print(f"  {'ECE':<20} {model_ece:>10.4f} {market_ece:>10.4f} {'Model' if model_ece < market_ece else 'Market':>10}")
        print(f"  {'ROI':<20} {model_roi:>9.1%} {market_roi:>9.1%} {'Model' if model_roi > market_roi else 'Market':>10}")

        all_results.append({
            "league": test_names[league],
            "n_matches": len(y_true),
            "model_ll": model_ll,
            "market_ll": market_ll,
            "model_brier": model_brier,
            "market_brier": market_brier,
            "model_acc": model_acc,
            "market_acc": market_acc,
            "model_ece": model_ece,
            "market_ece": market_ece,
            "model_roi": model_roi,
            "market_roi": market_roi,
        })

    # Summary
    if all_results:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        results_df.to_csv("backtests/results/real_odds_market_comparison.csv", index=False)
        print(f"\nResults saved to backtests/results/real_odds_market_comparison.csv")


def simulate_roi(probs, y_true, odds_list, strategy="model"):
    """Simulate ROI with flat staking."""
    bankroll = 1000.0
    start = bankroll
    stake = 10.0

    for i in range(len(y_true)):
        pred = np.argmax(probs[i])
        actual = y_true[i]

        # Get odds for predicted outcome
        if pred == 0:  # home
            odds = odds_list[i]["home"]
        elif pred == 1:  # draw
            odds = odds_list[i]["draw"]
        else:  # away
            odds = odds_list[i]["away"]

        if pd.isna(odds) or odds <= 1.0:
            continue

        # Check edge (model prob > market implied prob)
        market_prob = 1.0 / odds
        model_prob = probs[i][pred]

        if strategy == "model":
            # Only bet if model has edge
            if model_prob <= market_prob:
                continue

        if actual == pred:
            bankroll += stake * (odds - 1)
        else:
            bankroll -= stake

    return (bankroll - start) / start


if __name__ == "__main__":
    run_market_comparison()
