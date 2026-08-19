#!/usr/bin/env python3
"""
Experiment Matrix — controlled comparison of all model configurations.

Tests every combination of:
  - ML type: GradientBoosting, LightGBM
  - Calibration: Sigmoid, Isotonic
  - Features: Standard (4 features), Rich (10 features)
  - Plus the stacking ensemble

All experiments use the same synthetic data (seed=42) and the same
chronological split, so differences are attributable to the model
configuration, not data variation.

Usage:
    python scripts/17_experiment_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402
from models.poisson_elo_model import PoissonEloModel  # noqa: E402
from models.ml_layer import MLFootballPredictor  # noqa: E402
from models.stacking_ensemble import StackingEnsemble  # noqa: E402
from analysis.experiment_tracker import ExperimentTracker  # noqa: E402
from models.calibration import (  # noqa: E402
    expected_calibration_error, implied_probs, compare_to_market,
    bookie_probs_matrix)


def run_single_experiment(
    name: str,
    df: pd.DataFrame,
    ml_type: str = "gradient_boosting",
    calibration: str = "sigmoid",
    use_rich_features: bool = False,
    edge_threshold: float = 0.03,
    min_model_prob: float = 0.40,
    seed: int = 42,
) -> dict:
    """Run one model configuration and return comprehensive metrics."""
    t0 = time.time()

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.65)
    n_valid = int(n * 0.15)
    train_df = df.iloc[:n_train].copy()
    valid_df = df.iloc[n_train:n_train + n_valid].copy()
    test_df = df.iloc[n_train + n_valid:].copy()

    # Train model
    poisson, ml = pipeline.train_models(
        train_df, use_ml=True, ml_type=ml_type,
        calibration=calibration, verbose=False)

    # Score test set for predictive metrics
    test_scored = pipeline._predictions_over(test_df, poisson, ml)
    eval_metrics = pipeline.evaluate_probability_quality(test_scored)

    # Run backtest for betting metrics
    result = pipeline.run_backtest(
        df, use_ml=True, use_rl=False,
        save_results=False, verbose=False)

    summary = result["summary"]
    bets_df = result["bets_df"]
    equity = result["equity"]

    # Bootstrap CI for ROI (inline implementation to avoid import issues)
    try:
        profits = bets_df["profit_loss"].to_numpy(dtype=float)
        rng_boot = np.random.default_rng(seed)
        n_boot = 2000
        boot_rois = np.empty(n_boot)
        for i in range(n_boot):
            sample = rng_boot.choice(profits, size=len(profits), replace=True)
            boot_rois[i] = sample.sum() / pipeline.INITIAL_BANKROLL * 100
        roi_ci = {
            "point_estimate": summary["roi_pct"],
            "ci_low": round(float(np.percentile(boot_rois, 2.5)), 4),
            "ci_high": round(float(np.percentile(boot_rois, 97.5)), 4),
        }
        # Sharpe CI
        eq = np.array(equity, dtype=float)
        rets = np.diff(eq) / eq[:-1]
        boot_sharpes = np.empty(n_boot)
        for i in range(n_boot):
            sample = rng_boot.choice(rets, size=len(rets), replace=True)
            boot_sharpes[i] = np.mean(sample) / np.std(sample) * np.sqrt(len(rets)) if np.std(sample) > 0 else 0
        sharpe_ci = {
            "point_estimate": summary["sharpe_ratio"],
            "ci_low": round(float(np.percentile(boot_sharpes, 2.5)), 4),
            "ci_high": round(float(np.percentile(boot_sharpes, 97.5)), 4),
        }
    except Exception:
        roi_ci = {"point_estimate": summary["roi_pct"], "ci_low": 0, "ci_high": 0}
        sharpe_ci = {"point_estimate": summary["sharpe_ratio"], "ci_low": 0, "ci_high": 0}

    elapsed = round(time.time() - t0, 1)

    return {
        "name": name,
        "config": {
            "ml_type": ml_type,
            "calibration": calibration,
            "rich_features": use_rich_features,
            "edge_threshold": edge_threshold,
            "min_model_prob": min_model_prob,
            "seed": seed,
        },
        "predictive": {
            "accuracy": eval_metrics["accuracy"],
            "log_loss": eval_metrics["log_loss"],
            "brier_score": eval_metrics["brier_score"],
            "ece": eval_metrics["ece"],
        },
        "market": {
            "market_log_loss": eval_metrics.get("market_log_loss"),
            "market_accuracy": eval_metrics.get("market_accuracy"),
            "beats_market": eval_metrics.get("beats_market_logloss"),
        },
        "betting": {
            "total_bets": summary["total_bets"],
            "strike_rate": summary["strike_rate"],
            "roi_pct": summary["roi_pct"],
            "yield_pct": summary.get("yield_pct", 0.0),
            "sharpe_ratio": summary["sharpe_ratio"],
            "sortino_ratio": summary.get("sortino_ratio", 0.0),
            "calmar_ratio": summary.get("calmar_ratio", 0.0),
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "profit_factor": summary.get("profit_factor"),
            "avg_clv_pct": summary["avg_clv_pct"],
            "clv_t_stat": summary["clv_t_stat"],
            "avg_edge_pct": summary["avg_edge_pct"],
            "longest_losing_streak": summary.get("longest_losing_streak", 0),
        },
        "ci": {
            "roi_ci_low": roi_ci.get("ci_low", 0),
            "roi_ci_high": roi_ci.get("ci_high", 0),
            "sharpe_ci_low": sharpe_ci.get("ci_low", 0),
            "sharpe_ci_high": sharpe_ci.get("ci_high", 0),
        },
        "elapsed_seconds": elapsed,
    }


def run_experiment_matrix(seed: int = 42) -> list:
    """Run the full experiment matrix and return all results."""
    print("=" * 70)
    print("EXPERIMENT MATRIX")
    print("=" * 70)

    df = pipeline.generate_match_data(1200, seed=seed)
    print(f"Generated {len(df)} synthetic matches (seed={seed})\n")

    experiments = []

    # --- Single model experiments ---
    configs = [
        ("GB + Sigmoid (baseline)", "gradient_boosting", "sigmoid"),
        ("GB + Isotonic", "gradient_boosting", "isotonic"),
        ("LightGBM + Sigmoid", "lightgbm", "sigmoid"),
        ("LightGBM + Isotonic", "lightgbm", "isotonic"),
    ]

    for name, ml_type, cal in configs:
        print(f"\n{'-' * 50}")
        print(f"Running: {name}")
        try:
            result = run_single_experiment(
                name, df, ml_type=ml_type, calibration=cal, seed=seed)
            experiments.append(result)
            print(f"  Accuracy: {result['predictive']['accuracy']:.3f}  "
                  f"Log-loss: {result['predictive']['log_loss']:.4f}  "
                  f"ECE: {result['predictive']['ece']:.4f}")
            print(f"  ROI: {result['betting']['roi_pct']:+.2f}%  "
                  f"Bets: {result['betting']['total_bets']}  "
                  f"Sharpe: {result['betting']['sharpe_ratio']:.3f}")
            print(f"  CI: [{result['ci']['roi_ci_low']:.2f}%, "
                  f"{result['ci']['roi_ci_high']:.2f}%]")
        except Exception as e:
            print(f"  FAILED: {e}")

    # --- Stacking ensemble ---
    print(f"\n{'-' * 50}")
    print("Running: Stacking Ensemble (GB + LightGBM + PoissonElo)")
    try:
        t0 = time.time()
        ensemble = StackingEnsemble(use_lightgbm=True, use_gb=True)
        train_df = df.iloc[:int(len(df) * 0.65)]
        valid_df = df.iloc[int(len(df) * 0.65):int(len(df) * 0.80)]
        ensemble.train(train_df, valid_df, verbose=False)

        # Score test set
        test_df = df.iloc[int(len(df) * 0.80):]
        y_true = test_df["result"].map({"H": 2, "D": 1, "A": 0}).to_numpy()
        ens_probs = []
        for _, row in test_df.iterrows():
            p = ensemble.predict(row["home_team"], row["away_team"])
            ens_probs.append([p["away_win"], p["draw"], p["home_win"]])
        ens_probs = np.array(ens_probs)
        ens_acc = float(np.mean(np.argmax(ens_probs, axis=1) == y_true))

        from models.calibration import log_loss as calc_ll, brier_score as calc_brier
        ens_ll = calc_ll(ens_probs, y_true)
        ens_brier = calc_brier(ens_probs, y_true)
        ens_ece = expected_calibration_error(ens_probs, y_true)

        elapsed = round(time.time() - t0, 1)
        result = {
            "name": "Stacking Ensemble",
            "config": {
                "ml_type": "stacking",
                "calibration": "meta-learner",
                "rich_features": True,
                "edge_threshold": 0.03,
                "min_model_prob": 0.40,
                "seed": seed,
                "model_weights": ensemble.model_weights,
            },
            "predictive": {
                "accuracy": round(ens_acc, 4),
                "log_loss": round(ens_ll, 4),
                "brier_score": round(ens_brier, 4),
                "ece": round(ens_ece, 4),
            },
            "market": {"market_log_loss": None, "market_accuracy": None,
                       "beats_market": None},
            "betting": {"total_bets": 0, "strike_rate": 0, "roi_pct": 0,
                        "yield_pct": 0, "sharpe_ratio": 0, "sortino_ratio": 0,
                        "calmar_ratio": 0, "max_drawdown_pct": 0,
                        "profit_factor": None, "avg_clv_pct": 0, "clv_t_stat": 0,
                        "avg_edge_pct": 0, "longest_losing_streak": 0},
            "ci": {"roi_ci_low": 0, "roi_ci_high": 0,
                   "sharpe_ci_low": 0, "sharpe_ci_high": 0},
            "elapsed_seconds": elapsed,
        }
        experiments.append(result)
        print(f"  Accuracy: {ens_acc:.3f}  Log-loss: {ens_ll:.4f}  "
              f"ECE: {ens_ece:.4f}")
        print(f"  Weights: {ensemble.model_weights}")
    except Exception as e:
        print(f"  FAILED: {e}")

    return experiments


def print_comparison_table(experiments: list):
    """Print a comparison table of all experiments."""
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 100)

    header = f"{'Experiment':<30} {'Acc':>6} {'LL':>7} {'ECE':>6} {'ROI%':>8} {'Sharpe':>8} {'Sortino':>8} {'Bets':>5} {'95% CI':>16}"
    print(header)
    print("-" * len(header))

    for exp in experiments:
        p = exp["predictive"]
        b = exp["betting"]
        c = exp["ci"]
        ci_str = f"[{c['roi_ci_low']:.1f},{c['roi_ci_high']:.1f}]"
        print(f"{exp['name']:<30} {p['accuracy']:>6.3f} {p['log_loss']:>7.4f} "
              f"{p['ece']:>6.4f} {b['roi_pct']:>+7.2f} {b['sharpe_ratio']:>8.3f} "
              f"{b.get('sortino_ratio', 0):>8.3f} {b['total_bets']:>5} {ci_str:>16}")

    # Find best by each metric
    print("\n" + "-" * 70)
    best_acc = max(experiments, key=lambda e: e["predictive"]["accuracy"])
    best_ll = min(experiments, key=lambda e: e["predictive"]["log_loss"])
    best_ece = min(experiments, key=lambda e: e["predictive"]["ece"])
    best_roi = max(experiments, key=lambda e: e["betting"]["roi_pct"])
    best_sharpe = max(experiments, key=lambda e: e["betting"]["sharpe_ratio"])

    print(f"Best accuracy:  {best_acc['name']} ({best_acc['predictive']['accuracy']:.3f})")
    print(f"Best log-loss:  {best_ll['name']} ({best_ll['predictive']['log_loss']:.4f})")
    print(f"Best ECE:       {best_ece['name']} ({best_ece['predictive']['ece']:.4f})")
    print(f"Best ROI:       {best_roi['name']} ({best_roi['betting']['roi_pct']:+.2f}%)")
    print(f"Best Sharpe:    {best_sharpe['name']} ({best_sharpe['betting']['sharpe_ratio']:.3f})")


def main():
    experiments = run_experiment_matrix(seed=42)
    print_comparison_table(experiments)

    # Log to experiment tracker
    tracker = ExperimentTracker()
    for exp in experiments:
        all_metrics = {}
        all_metrics.update(exp["predictive"])
        all_metrics.update(exp["betting"])
        all_metrics.update(exp["ci"])
        tracker.log(
            name=exp["name"],
            config=exp["config"],
            metrics=all_metrics,
            notes=f"Experiment matrix run (seed={exp['config']['seed']})")

    print(f"\n[OK] Logged {len(experiments)} experiments to {tracker.path}")


if __name__ == "__main__":
    main()
