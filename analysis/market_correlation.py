#!/usr/bin/env python3
"""
Market Correlation Analysis — measures how correlated model predictions
are with bookmaker implied probabilities.

Key questions answered:
1. Is the model merely reproducing the market?
2. Do profitable predictions occur where the model disagrees with the market?
3. What is the distribution of probability differences?

Usage:
    from analysis.market_correlation import MarketCorrelationAnalyzer
    analyzer = MarketCorrelationAnalyzer()
    report = analyzer.analyze(df, model_probs, market_probs)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MarketCorrelationAnalyzer:
    """Analyze the relationship between model predictions and market odds."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def implied_probabilities(self, odds_home: np.ndarray,
                              odds_draw: np.ndarray,
                              odds_away: np.ndarray) -> np.ndarray:
        """Convert decimal odds to normalized implied probabilities.

        Returns (n, 3) array [away, draw, home].
        """
        inv_home = 1.0 / np.clip(odds_home, 1.01, 100)
        inv_draw = 1.0 / np.clip(odds_draw, 1.01, 100)
        inv_away = 1.0 / np.clip(odds_away, 1.01, 100)
        total = inv_home + inv_draw + inv_away
        return np.column_stack([inv_away / total, inv_draw / total,
                                inv_home / total])

    def analyze(self, df: pd.DataFrame,
                model_probs: np.ndarray,
                odds_cols: Tuple[str, str, str] = (
                    "odds_home_b365", "odds_draw_b365", "odds_away_b365")
                ) -> Dict:
        """Full market correlation analysis.

        Args:
            df: DataFrame with odds columns.
            model_probs: (n, 3) model probabilities [away, draw, home].
            odds_cols: column names for home/draw/away odds.

        Returns:
            Dict with correlation stats, EV distribution, and edge analysis.
        """
        # Get market implied probabilities
        market_probs = self.implied_probabilities(
            df[odds_cols[0]].values,
            df[odds_cols[1]].values,
            df[odds_cols[2]].values
        )

        # Ensure same shape
        n = min(len(model_probs), len(market_probs))
        model_p = model_probs[:n]
        market_p = market_probs[:n]

        result = {}

        # 1. Pearson correlation per outcome
        outcome_names = ["away_win", "draw", "home_win"]
        correlations = {}
        for i, name in enumerate(outcome_names):
            r, p = stats.pearsonr(model_p[:, i], market_p[:, i])
            correlations[name] = {
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 6),
            }
        result["correlations"] = correlations

        # 2. Spearman rank correlation
        spearman_correlations = {}
        for i, name in enumerate(outcome_names):
            rho, p = stats.spearmanr(model_p[:, i], market_p[:, i])
            spearman_correlations[name] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p), 6),
            }
        result["spearman_correlations"] = spearman_correlations

        # 3. Probability difference (model - market)
        prob_diffs = model_p - market_p
        result["probability_difference"] = {}
        for i, name in enumerate(outcome_names):
            diffs = prob_diffs[:, i]
            result["probability_difference"][name] = {
                "mean": round(float(diffs.mean()), 4),
                "std": round(float(diffs.std()), 4),
                "min": round(float(diffs.min()), 4),
                "max": round(float(diffs.max()), 4),
                "mean_abs": round(float(np.abs(diffs).mean()), 4),
            }

        # 4. Expected value distribution
        # For each outcome: EV = model_prob * odds - 1
        odds_matrix = np.column_stack([
            df[odds_cols[2]].values[:n],  # home
            df[odds_cols[1]].values[:n],  # draw
            df[odds_cols[0]].values[:n],  # away
        ])
        ev_matrix = model_p * odds_matrix - 1.0
        max_ev = ev_matrix.max(axis=1)

        result["ev_distribution"] = {
            "mean_max_ev": round(float(max_ev.mean()), 4),
            "std_max_ev": round(float(max_ev.std()), 4),
            "positive_ev_fraction": round(float((max_ev > 0).mean()), 4),
            "mean_ev_home": round(float(ev_matrix[:, 0].mean()), 4),
            "mean_ev_draw": round(float(ev_matrix[:, 1].mean()), 4),
            "mean_ev_away": round(float(ev_matrix[:, 2].mean()), 4),
        }

        # 5. Model vs market edge analysis
        # Where model disagrees most with market
        disagreement = np.abs(prob_diffs).max(axis=1)
        top_quartile = disagreement > np.percentile(disagreement, 75)
        bottom_quartile = disagreement < np.percentile(disagreement, 25)

        result["disagreement_analysis"] = {
            "mean_disagreement": round(float(disagreement.mean()), 4),
            "median_disagreement": round(float(np.median(disagreement)), 4),
            "top_quartile_mean_disagreement": round(
                float(disagreement[top_quartile].mean()), 4),
            "bottom_quartile_mean_disagreement": round(
                float(disagreement[bottom_quartile].mean()), 4),
        }

        # 6. Market overround
        overrounds = (1 / df[odds_cols[0]].values[:n] +
                      1 / df[odds_cols[1]].values[:n] +
                      1 / df[odds_cols[2]].values[:n]) - 1.0
        result["overround"] = {
            "mean": round(float(overrounds.mean()), 4),
            "std": round(float(overrounds.std()), 4),
            "min": round(float(overrounds.min()), 4),
            "max": round(float(overrounds.max()), 4),
        }

        # 7. Classification: where model beats market
        y_true = df["result"].values[:n]
        model_pred = np.argmax(model_p, axis=1)
        market_pred = np.argmax(market_p, axis=1)
        result_class = {"H": 2, "D": 1, "A": 0}
        y_idx = np.array([result_class.get(r, 1) for r in y_true])

        model_correct = model_pred == y_idx
        market_correct = market_pred == y_idx

        result["head_to_head"] = {
            "model_accuracy": round(float(model_correct.mean()), 4),
            "market_accuracy": round(float(market_correct.mean()), 4),
            "model_beats_market_count": int((model_correct & ~market_correct).sum()),
            "market_beats_model_count": int((market_correct & ~model_correct).sum()),
            "both_correct_count": int((model_correct & market_correct).sum()),
            "both_wrong_count": int((~model_correct & ~market_correct).sum()),
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _print_report(self, result: Dict):
        """Pretty-print the analysis report."""
        print("\n" + "=" * 60)
        print("MARKET CORRELATION ANALYSIS")
        print("=" * 60)

        print("\nCorrelations (model vs market):")
        for name, corr in result["correlations"].items():
            sp = result["spearman_correlations"][name]
            print(f"  {name}: Pearson r={corr['pearson_r']:.3f} "
                  f"(p={corr['p_value']:.4f}), "
                  f"Spearman rho={sp['spearman_rho']:.3f}")

        print("\nProbability differences (model - market):")
        for name, diff in result["probability_difference"].items():
            print(f"  {name}: mean={diff['mean']:+.4f}, "
                  f"mean|diff|={diff['mean_abs']:.4f}")

        print(f"\nOverround: {result['overround']['mean']:.4f} "
              f"(~{result['overround']['mean']*100:.1f}%)")

        h2h = result["head_to_head"]
        print(f"\nHead-to-head:")
        print(f"  Model accuracy:  {h2h['model_accuracy']:.3f}")
        print(f"  Market accuracy: {h2h['market_accuracy']:.3f}")
        print(f"  Model beats market: {h2h['model_beats_market_count']} times")
        print(f"  Market beats model: {h2h['market_beats_model_count']} times")

        ev = result["ev_distribution"]
        print(f"\nEV distribution:")
        print(f"  Mean max EV: {ev['mean_max_ev']:+.4f}")
        print(f"  Fraction positive EV: {ev['positive_ev_fraction']:.3f}")


if __name__ == "__main__":
    import pipeline
    df = pipeline.generate_match_data(600, seed=42)
    from models.poisson_elo_model import PoissonEloModel
    from models.ml_layer import MLFootballPredictor

    poisson = PoissonEloModel(use_dixon_coles=False)
    poisson.train(df.iloc[:420])
    ml = MLFootballPredictor()
    ml.train(poisson.training_features, verbose=False)

    test = df.iloc[480:]
    model_probs = np.zeros((len(test), 3))
    for i, (_, row) in enumerate(test.iterrows()):
        p = pipeline.ensemble_probs(poisson, ml, row["home_team"],
                                    row["away_team"])
        model_probs[i] = [p["away_win"], p["draw"], p["home_win"]]

    analyzer = MarketCorrelationAnalyzer(verbose=True)
    report = analyzer.analyze(test, model_probs)
    print("\n[OK] Market correlation analysis complete.")
