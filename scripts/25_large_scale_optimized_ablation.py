#!/usr/bin/env python3
"""
Large-Scale Optimized Ablation — 107K Matches.

Uses fast KDE, vectorized MC, and online Poisson ratings.
Tests each layer incrementally on a large dataset.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from data.real_data import load_league
from models.layered_model import LayeredModel


def run_optimized_ablation(n_matches: int = 10000, verbose: bool = True):
    """Run ablation tournament with optimized components."""
    if verbose:
        print("=" * 70)
        print("OPTIMIZED LAYERED MODEL ABLATION")
        print("=" * 70)

    # Load data from multiple leagues
    print("Loading data...")
    leagues = ['SP1', 'E0', 'D1', 'I1', 'F1']
    dfs = []
    for league in leagues:
        try:
            league_df = load_league(league, seasons=None, offline=True)
            if len(league_df) > 0:
                dfs.append(league_df)
                print(f"  {league}: {len(league_df)} matches")
        except Exception as e:
            print(f"  {league}: Failed ({e})")
    
    if not dfs:
        print("No data loaded!")
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) > n_matches:
        df = df.tail(n_matches).reset_index(drop=True)
    print(f"Total: {len(df)} matches")

    # Split chronologically
    n = len(df)
    split_idx = int(n * 0.6)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Configurations
    configs = [
        ("Baseline (Poisson only)", []),
        ("+ Bayesian", ["bayesian"]),
        ("+ EWMA", ["bayesian", "ewma"]),
        ("+ Fast KDE", ["bayesian", "ewma", "kde"]),
        ("+ Vectorized MC", ["bayesian", "ewma", "kde", "mixture_mc"]),
        ("+ Contextual", ["bayesian", "ewma", "kde", "mixture_mc", "contextual"]),
        ("+ Online Poisson", ["bayesian", "ewma", "kde", "mixture_mc", "contextual", "online_poisson"]),
        ("Full Ensemble", ["bayesian", "ewma", "kde", "mixture_mc", "contextual", "online_poisson", "ensemble"]),
    ]

    results = []

    for config_name, active_layers in configs:
        if verbose:
            print(f"\n--- {config_name} ---")

        t0 = time.perf_counter()

        # Train
        model = LayeredModel(layers=active_layers)
        model.train(train_df, verbose=False)

        # Test
        all_probs = []
        all_true = []

        for _, row in test_df.iterrows():
            try:
                probs = model.predict(row["home_team"], row["away_team"])
                all_probs.append([probs["away_win"], probs["draw"], probs["home_win"]])

                true_map = {"A": 0, "D": 1, "H": 2}
                all_true.append(true_map.get(row["result"], 1))
            except Exception:
                continue

        t_total = (time.perf_counter() - t0) * 1000

        if not all_probs:
            continue

        probs_arr = np.array(all_probs)
        y_true = np.array(all_true)

        # Metrics
        eps = 1e-9
        log_loss = float(-np.mean(np.log(np.clip(probs_arr[np.arange(len(y_true)), y_true], eps, 1))))
        brier = float(np.mean(np.sum((probs_arr - np.eye(3)[y_true]) ** 2, axis=1)))
        accuracy = float(np.mean(np.argmax(probs_arr, axis=1) == y_true))

        # ECE
        from models.calibration import expected_calibration_error
        ece = expected_calibration_error(probs_arr, y_true)

        result = {
            "config": config_name,
            "n_layers": len(active_layers),
            "log_loss": round(log_loss, 4),
            "brier": round(brier, 4),
            "accuracy": round(accuracy, 4),
            "ece": round(ece, 4),
            "time_ms": round(t_total, 0),
            "n_test": len(y_true),
        }
        results.append(result)

        if verbose:
            print(f"  Log-loss: {log_loss:.4f} | Brier: {brier:.4f} | "
                  f"Acc: {accuracy:.1%} | ECE: {ece:.4f} | Time: {t_total:.0f}ms")

    results_df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION RESULTS")
        print("=" * 70)
        print(results_df.to_string(index=False))

        # Save results
        results_df.to_csv("backtests/results/large_scale_optimized_ablation.csv", index=False)
        print(f"\nResults saved to backtests/results/large_scale_optimized_ablation.csv")

    return results_df


if __name__ == "__main__":
    run_optimized_ablation(n_matches=10000, verbose=True)
