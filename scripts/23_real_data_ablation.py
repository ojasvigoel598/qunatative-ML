#!/usr/bin/env python3
"""
Real-Data Ablation — test layered model on real La Liga data.

On synthetic Poisson data, the simple model wins (honest finding).
On real data, the distributions are non-Poisson and regimes change,
so the layered system should add value.

Usage:
    python scripts/23_real_data_ablation.py --offline
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.real_data import load_league, LEAGUES, SEASON_CODES, SEASON_LABEL


def prepare_real_data(league="SP1", offline=True):
    """Load and prepare real La Liga data."""
    frames = []
    for s in SEASON_CODES:
        try:
            frames.append(load_league(league, [s], offline=offline))
        except Exception:
            continue

    if not frames:
        print("No data loaded")
        return None

    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} matches from {LEAGUES[league]}")
    return df


def run_real_ablation(df, verbose=True):
    """Run ablation on real data."""
    from models.layered_model import LayeredModel

    if verbose:
        print("=" * 70)
        print("REAL-DATA ABLATION TOURNAMENT")
        print("=" * 70)

    df = df.sort_values("date").reset_index(drop=True)

    # Walk-forward: train on first 60%, test on last 40%
    n = len(df)
    split_idx = int(n * 0.6)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train: {len(train_df)} matches ({train_df['date'].iloc[0].date()} -> {train_df['date'].iloc[-1].date()})")
    print(f"Test:  {len(test_df)} matches ({test_df['date'].iloc[0].date()} -> {test_df['date'].iloc[-1].date()})")

    configs = [
        ("Baseline (Poisson only)", []),
        ("+ Bayesian", ["bayesian"]),
        ("+ EWMA", ["bayesian", "ewma"]),
        ("+ KDE", ["bayesian", "ewma", "kde"]),
        ("+ Mixture MC", ["bayesian", "ewma", "kde", "mixture_mc"]),
        ("+ Contextual", ["bayesian", "ewma", "kde", "mixture_mc", "contextual"]),
        ("Full Ensemble", ["bayesian", "ewma", "kde", "mixture_mc", "contextual", "ensemble"]),
    ]

    results = []

    for config_name, active_layers in configs:
        if verbose:
            print(f"\n--- {config_name} ---")

        model = LayeredModel(layers=active_layers)
        model.train(train_df, verbose=False)

        all_probs = []
        all_true = []
        all_bookie = []

        for _, row in test_df.iterrows():
            try:
                probs = model.predict(row["home_team"], row["away_team"])
                all_probs.append([probs["away_win"], probs["draw"], probs["home_win"]])

                true_map = {"A": 0, "D": 1, "H": 2}
                all_true.append(true_map.get(row["result"], 1))

                # Market odds for comparison
                if "odds_home" in row and pd.notna(row.get("odds_home")):
                    all_bookie.append([
                        1.0 / row.get("odds_away", 3.0),
                        1.0 / row.get("odds_draw", 3.5),
                        1.0 / row.get("odds_home", 2.0),
                    ])
            except Exception:
                continue

        if not all_probs:
            continue

        probs_arr = np.array(all_probs)
        y_true = np.array(all_true)

        eps = 1e-9
        log_loss = float(-np.mean(np.log(np.clip(probs_arr[np.arange(len(y_true)), y_true], eps, 1))))
        brier = float(np.mean(np.sum((probs_arr - np.eye(3)[y_true]) ** 2, axis=1)))
        accuracy = float(np.mean(np.argmax(probs_arr, axis=1) == y_true))

        from models.calibration import expected_calibration_error
        ece = expected_calibration_error(probs_arr, y_true)

        # Market comparison
        market_ll = None
        if all_bookie:
            bookie_arr = np.array(all_bookie)
            bookie_arr = bookie_arr / bookie_arr.sum(axis=1, keepdims=True)
            market_ll = float(-np.mean(np.log(np.clip(bookie_arr[np.arange(len(y_true)), y_true], eps, 1))))

        result = {
            "config": config_name,
            "n_layers": len(active_layers),
            "log_loss": round(log_loss, 4),
            "brier": round(brier, 4),
            "accuracy": round(accuracy, 4),
            "ece": round(ece, 4),
            "market_log_loss": round(market_ll, 4) if market_ll else None,
            "beats_market": bool(log_loss < market_ll) if market_ll else None,
            "n_test": len(y_true),
        }
        results.append(result)

        if verbose:
            market_str = f" | Market LL: {market_ll:.4f}" if market_ll else ""
            beats = "YES" if result["beats_market"] else "NO" if result["beats_market"] is not None else "?"
            print(f"  LL: {log_loss:.4f} | Brier: {brier:.4f} | "
                  f"Acc: {accuracy:.1%} | ECE: {ece:.4f}{market_str} | Beats: {beats}")

    results_df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 70)
        print("REAL-DATA ABLATION RESULTS")
        print("=" * 70)
        print(results_df.to_string(index=False))

        # Find best
        best_ll = results_df.loc[results_df["log_loss"].idxmin()]
        print(f"\nBest log-loss: {best_ll['config']} ({best_ll['log_loss']:.4f})")

        # Check if any model beats the market
        beats_market = results_df[results_df["beats_market"] == True]
        if len(beats_market) > 0:
            print(f"\nModels that beat the market:")
            for _, row in beats_market.iterrows():
                print(f"  {row['config']}: LL={row['log_loss']:.4f} vs Market={row['market_log_loss']:.4f}")
        else:
            print(f"\nNo model beats the market on this sample.")

    return results_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-Data Ablation")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--league", default="SP1", choices=["SP1", "E0", "I1"])
    args = parser.parse_args()

    df = prepare_real_data(league=args.league, offline=args.offline)
    if df is not None:
        results = run_real_ablation(df, verbose=True)
        results.to_csv(PROJECT_ROOT / "backtests" / "results" / "real_ablation_results.csv", index=False)
        print(f"\n[OK] Results saved to backtests/results/real_ablation_results.csv")


if __name__ == "__main__":
    main()
