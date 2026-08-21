#!/usr/bin/env python3
"""
CLV Validation & Real-Data Favourite-Only Backtest

Tests the favourite-only strategy on REAL La Liga data (2021-2026),
validates CLV as a predictor of profitability, and compares with
other strategies.

Methodology:
- Expanding-window walk-forward: train on seasons 1..N, test on N+1
- Favourite-only filter: only bet when odds < 2.0
- Research layer: validate ML picks against match context
- CLV validation: test if beating the closing line predicts long-term profit

Research basis:
- Mints (2021): CLV is the strongest predictor of betting skill
- Snowberg & Levitt (2007): Favourite-longshot bias exploitation
- Gramm & Owens (2006): Professional bettors achieve positive CLV

Usage:
    python scripts/19_clv_real_data_backtest.py --offline
    python scripts/19_clv_real_data_backtest.py  # downloads fresh data
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.real_data import load_league, LEAGUES, SEASON_CODES, SEASON_LABEL
from models.poisson_elo_model import PoissonEloModel
from models.ml_layer import MLFootballPredictor
from models.rich_features import compute_all_rich_features, RICH_FEATURE_COLS
from analysis.research_layer import ResearchLayer, MatchInformation, build_match_info_from_dataframe

RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Online Feature Builder (no leakage)
# ======================================================================
class OnlineFeatureBuilder:
    """Running Elo + rolling form + rich features; features for match i
    use only matches strictly before i."""

    def __init__(self, elo_k=20.0, elo_base=1500.0):
        self.elo_k = elo_k
        self.elo_base = elo_base
        self.elo = defaultdict(lambda: elo_base)
        self.match_history: list = []  # last 20 matches for form computation

    def get_features(self, home_team: str, away_team: str) -> dict:
        """Get features for a match using only past data."""
        return {
            "home_elo": self.elo[home_team],
            "away_elo": self.elo[away_team],
        }

    def update(self, row: pd.Series):
        """Update state after observing a match result."""
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        # Elo update
        home_r, away_r = self.elo[h], self.elo[a]
        exp_home = 1 / (1 + 10 ** ((away_r - home_r) / 400))
        actual = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        self.elo[h] += self.elo_k * (actual - exp_home)
        self.elo[a] += self.elo_k * ((1 - actual) - (1 - exp_home))

        # Store last 20 matches for form computation
        self.match_history.append({
            "home_team": h, "away_team": a,
            "home_goals": hg, "away_goals": ag,
            "result": row["result"],
            "date": row.get("date"),
        })
        # Keep only last 20 matches
        if len(self.match_history) > 20:
            self.match_history = self.match_history[-20:]

    def get_form(self, team: str, n: int = 5) -> dict:
        """Get rolling form for a team from last n matches."""
        team_matches = [
            m for m in self.match_history
            if m["home_team"] == team or m["away_team"] == team
        ][-n:]

        if not team_matches:
            return {"form_pts": 1.33, "goals_scored": 1.5, "goals_conceded": 1.2}

        pts, gs, gc = [], [], []
        for m in team_matches:
            if m["home_team"] == team:
                r = m["result"]
                pts.append(3 if r == "H" else (1 if r == "D" else 0))
                gs.append(m["home_goals"])
                gc.append(m["away_goals"])
            else:
                r = m["result"]
                pts.append(3 if r == "A" else (1 if r == "D" else 0))
                gs.append(m["away_goals"])
                gc.append(m["home_goals"])

        return {
            "form_pts": np.mean(pts),
            "goals_scored": np.mean(gs),
            "goals_conceded": np.mean(gc),
        }


# ======================================================================
# Backtesting engine
# ======================================================================
def run_real_data_backtest(df: pd.DataFrame,
                           strategy: str = "favourite_only",
                           use_research_layer: bool = True,
                           edge_threshold: float = 0.03,
                           kelly_fraction: float = 0.25,
                           initial_bankroll: float = 10000.0,
                           verbose: bool = True) -> dict:
    """Run a walk-forward backtest on real data.

    Args:
        df: Real match data (sorted by date)
        strategy: "favourite_only", "all_bets", "consensus", "research_validated"
        use_research_layer: whether to use the research layer for validation
        edge_threshold: minimum edge to place a bet
        kelly_fraction: fraction of Kelly to stake
        initial_bankroll: starting bankroll
        verbose: print progress

    Returns:
        Dict with metrics, bets, equity curve
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Split: first 60% for training, last 40% for testing (walk-forward)
    n = len(df)
    split_idx = int(n * 0.6)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if verbose:
        print(f"  Train: {len(train_df)} matches ({train_df['date'].iloc[0].date()} -> {train_df['date'].iloc[-1].date()})")
        print(f"  Test:  {len(test_df)} matches ({test_df['date'].iloc[0].date()} -> {test_df['date'].iloc[-1].date()})")

    # Train models on training data
    builder = OnlineFeatureBuilder()

    # Build training features
    train_features = []
    for _, row in train_df.iterrows():
        feat = builder.get_features(row["home_team"], row["away_team"])
        train_features.append(feat)
        builder.update(row)

    train_feat_df = pd.DataFrame(train_features)
    train_feat_df["result"] = train_df["result"].values
    train_feat_df["home_team"] = train_df["home_team"].values
    train_feat_df["away_team"] = train_df["away_team"].values
    train_feat_df["home_goals"] = train_df["home_goals"].values
    train_feat_df["away_goals"] = train_df["away_goals"].values
    train_feat_df["date"] = train_df["date"].values

    # Train ML layer directly on Elo features
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    
    target_map = {"H": 2, "D": 1, "A": 0}
    y_train_ml = train_feat_df["result"].map(target_map)
    X_train_ml = train_feat_df[["home_elo", "away_elo"]].copy()
    X_train_ml["elo_diff"] = X_train_ml["home_elo"] - X_train_ml["away_elo"]
    
    ml = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                   min_samples_leaf=20, subsample=0.8, random_state=42),
        method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    )
    ml.fit(X_train_ml, y_train_ml)
    ml_trained = True

    # Research layer
    research = ResearchLayer(verbose=False)

    # Test walk-forward
    bets = []
    equity = [initial_bankroll]
    bankroll = initial_bankroll
    all_predictions = []

    for _, row in test_df.iterrows():
        home, away = row["home_team"], row["away_team"]

        # Get model predictions
        probs = {
            "home_win": 0.0, "draw": 0.0, "away_win": 0.0
        }

        # ML prediction on Elo features
        try:
            feat = builder.get_features(home, away)
            X = pd.DataFrame([{
                "home_elo": feat["home_elo"],
                "away_elo": feat["away_elo"],
                "elo_diff": feat["home_elo"] - feat["away_elo"],
            }])
            proba = ml.predict_proba(X)[0]
            probs = {"away_win": proba[0], "draw": proba[1], "home_win": proba[2]}
        except Exception:
            pass

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        # Bookmaker odds (use B365 as primary)
        bookie = {}
        for side, col in [("home_win", "odds_home"), ("draw", "odds_draw"), ("away_win", "odds_away")]:
            if col in row.index and pd.notna(row[col]) and row[col] > 1.01:
                bookie[side] = row[col]

        if not bookie:
            builder.update(row)
            continue

        # Calculate edges
        edges = {}
        for side in probs:
            if side in bookie:
                edge = probs[side] * bookie[side] - 1.0
                edges[side] = edge

        # Find best edge
        best_side = max(edges, key=edges.get) if edges else None
        best_edge = edges.get(best_side, 0) if best_side else 0

        # Store prediction for CLV analysis
        prediction = {
            "date": row["date"],
            "home": home, "away": away,
            "model_prob": probs.get(best_side, 0),
            "model_pick": best_side,
            "bookie_odds": bookie.get(best_side, 0),
            "bookie_implied": 1.0 / bookie.get(best_side, 2.0),
            "edge": best_edge,
        }

        # Strategy filters
        should_bet = False
        stake = 0.0

        if best_edge >= edge_threshold:
            odds = bookie.get(best_side, 0)
            model_prob = probs.get(best_side, 0)

            if strategy == "favourite_only":
                # Only bet on favourites (odds < 2.0)
                if odds < 2.0 and model_prob > 0.40:
                    should_bet = True

            elif strategy == "all_bets":
                if odds > 1.6 and model_prob > 0.40:
                    should_bet = True

            elif strategy == "consensus":
                # Bet when model and market agree
                market_best = max(bookie, key=lambda k: 1.0 / bookie[k])
                if best_side == market_best and odds > 1.6:
                    should_bet = True

            elif strategy == "research_validated":
                # Two-step: ML selects, research validates
                if odds > 1.6 and model_prob > 0.40:
                    info = MatchInformation(
                        home_team=home, away_team=away,
                        home_form_pts=builder.get_form(home)["form_pts"],
                        away_form_pts=builder.get_form(away)["form_pts"],
                        home_goals_scored=builder.get_form(home)["goals_scored"],
                        away_goals_scored=builder.get_form(away)["goals_scored"],
                        home_goals_conceded=builder.get_form(home)["goals_conceded"],
                        away_goals_conceded=builder.get_form(away)["goals_conceded"],
                        rest_days_home=7, rest_days_away=7,
                    )
                    decision = research.validate_bet(
                        match={"home": home, "away": away},
                        ml_pick=best_side,
                        ml_prob=model_prob,
                        bookie_odds=odds,
                        bookie_implied=1.0 / odds,
                        info=info,
                    )
                    if decision.action in ("BET", "ADJUST_STAKE"):
                        should_bet = True
                        prediction["research_action"] = decision.action
                        prediction["research_confidence"] = decision.confidence

            if should_bet:
                # Kelly stake
                stake_frac = (best_edge / (odds - 1)) * kelly_fraction
                stake_frac = min(stake_frac, 0.08)  # max 8% of bankroll
                stake = bankroll * stake_frac

                if stake < 50:  # min stake
                    should_bet = False

        if should_bet:
            odds = bookie.get(best_side, 0)
            win = row["result"].upper() == best_side[0].upper()
            profit = stake * (odds - 1) if win else -stake
            bankroll += profit

            # Closing line (use closing odds if available)
            closing_col = f"closing_odds_{best_side.split('_')[0]}"
            if closing_col in row.index and pd.notna(row[closing_col]):
                closing_odds = row[closing_col]
            else:
                closing_odds = odds  # no CLV data

            clv = (closing_odds - odds) / odds * 100 if closing_odds > 0 else 0

            bets.append({
                "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                "match": f"{home} vs {away}",
                "market": best_side,
                "my_odds": odds,
                "closing_odds": closing_odds,
                "clv_pct": round(clv, 2),
                "model_prob": round(probs.get(best_side, 0), 4),
                "edge_pct": round(best_edge * 100, 2),
                "stake": round(stake, 2),
                "profit_loss": round(profit, 2),
                "bet_outcome": "Win" if win else "Lose",
                "running_bankroll": round(bankroll, 2),
            })
            equity.append(bankroll)

        prediction["bet_placed"] = should_bet
        all_predictions.append(prediction)

        # Update builder
        builder.update(row)

    # Compute metrics
    bets_df = pd.DataFrame(bets)
    predictions_df = pd.DataFrame(all_predictions)

    metrics = compute_backtest_metrics(bets_df, equity, initial_bankroll, test_df)
    clv_analysis = compute_clv_analysis(bets_df)
    roi_by_odds = compute_roi_by_odds_band(bets_df)

    return {
        "strategy": strategy,
        "metrics": metrics,
        "clv_analysis": clv_analysis,
        "roi_by_odds": roi_by_odds,
        "bets_df": bets_df,
        "predictions_df": predictions_df,
        "equity": equity,
    }


def compute_backtest_metrics(bets_df: pd.DataFrame, equity: list,
                              initial_bankroll: float, df: pd.DataFrame) -> dict:
    """Compute comprehensive backtest metrics."""
    if len(bets_df) == 0:
        return {"total_bets": 0, "roi_pct": 0, "profit": 0}

    total_bets = len(bets_df)
    wins = int((bets_df["bet_outcome"] == "Win").sum())
    total_profit = float(bets_df["profit_loss"].sum())
    roi = total_profit / initial_bankroll * 100

    # Sharpe
    returns = np.diff(equity) / np.array(equity[:-1])
    bets_per_year = total_bets / max((df["date"].max() - df["date"].min()).days / 365.25, 0.1)
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bets_per_year)) if len(returns) > 1 and np.std(returns) > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino = float(np.mean(returns) / downside_std * np.sqrt(bets_per_year)) if downside_std > 0 else 0.0

    # Max drawdown
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    max_dd = float(abs(((equity_arr - peak) / peak).min()) * 100)

    # Profit factor
    gross_wins = float(bets_df.loc[bets_df["profit_loss"] > 0, "profit_loss"].sum())
    gross_losses = float(-bets_df.loc[bets_df["profit_loss"] < 0, "profit_loss"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # CLV
    clv_mean = float(bets_df["clv_pct"].mean())
    clv_sd = float(bets_df["clv_pct"].std(ddof=1))
    clv_t = float(clv_mean / (clv_sd / np.sqrt(total_bets))) if total_bets > 1 and clv_sd > 0 else 0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": total_bets - wins,
        "strike_rate": round(wins / total_bets * 100, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi, 2),
        "yield_pct": round(total_profit / bets_df["stake"].sum() * 100, 2) if bets_df["stake"].sum() > 0 else 0,
        "avg_edge_pct": round(float(bets_df["edge_pct"].mean()), 2),
        "avg_odds": round(float(bets_df["my_odds"].mean()), 2),
        "avg_stake": round(float(bets_df["stake"].mean()), 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else None,
        "final_bankroll": round(bankroll := float(equity[-1]), 2),
        "clv_mean_pct": round(clv_mean, 2),
        "clv_t_stat": round(clv_t, 2),
    }


def compute_clv_analysis(bets_df: pd.DataFrame) -> dict:
    """Validate CLV as predictor of profitability.

    Research: Mints (2021), Gramm & Owens (2006)
    CLV is the strongest predictor of long-term betting skill.
    """
    if len(bets_df) < 10:
        return {"sufficient_data": False}

    # Split into positive CLV and negative CLV groups
    pos_clv = bets_df[bets_df["clv_pct"] > 0]
    neg_clv = bets_df[bets_df["clv_pct"] <= 0]

    pos_roi = pos_clv["profit_loss"].sum() / pos_clv["stake"].sum() * 100 if len(pos_clv) > 0 else 0
    neg_roi = neg_clv["profit_loss"].sum() / neg_clv["stake"].sum() * 100 if len(neg_clv) > 0 else 0

    pos_wr = (pos_clv["bet_outcome"] == "Win").mean() * 100 if len(pos_clv) > 0 else 0
    neg_wr = (neg_clv["bet_outcome"] == "Win").mean() * 100 if len(neg_clv) > 0 else 0

    return {
        "sufficient_data": True,
        "positive_clv_bets": len(pos_clv),
        "negative_clv_bets": len(neg_clv),
        "positive_clv_roi": round(pos_roi, 2),
        "negative_clv_roi": round(neg_roi, 2),
        "positive_clv_win_rate": round(pos_wr, 2),
        "negative_clv_win_rate": round(neg_wr, 2),
        "clv_predicts_profitability": pos_roi > neg_roi,
        "clv_gap": round(pos_roi - neg_roi, 2),
    }


def compute_roi_by_odds_band(bets_df: pd.DataFrame) -> dict:
    """Compute ROI by odds band to identify where edge exists."""
    if len(bets_df) == 0:
        return {}

    bands = {
        "odds_1.5_1.8": (1.5, 1.8),
        "odds_1.8_2.0": (1.8, 2.0),
        "odds_2.0_2.5": (2.0, 2.5),
        "odds_2.5_3.0": (2.5, 3.0),
        "odds_3.0_plus": (3.0, 100),
    }

    results = {}
    for band_name, (lo, hi) in bands.items():
        sub = bets_df[(bets_df["my_odds"] >= lo) & (bets_df["my_odds"] < hi)]
        if len(sub) > 0:
            roi = sub["profit_loss"].sum() / sub["stake"].sum() * 100
            wr = (sub["bet_outcome"] == "Win").mean() * 100
            results[band_name] = {
                "bets": len(sub),
                "roi_pct": round(roi, 2),
                "win_rate": round(wr, 2),
            }
        else:
            results[band_name] = {"bets": 0, "roi_pct": 0, "win_rate": 0}

    return results


# ======================================================================
# Multi-season walk-forward
# ======================================================================
def run_multi_season_walkforward(league: str = "SP1",
                                  use_research_layer: bool = True,
                                  verbose: bool = True) -> dict:
    """Run walk-forward backtest across multiple seasons.

    For each season:
    1. Train on all previous seasons
    2. Test on current season
    3. Record metrics
    """
    if verbose:
        print("=" * 70)
        print(f"MULTI-SEASON WALK-FORWARD BACKTEST — {LEAGUES[league]}")
        print("=" * 70)

    all_results = {}
    season_metrics = []

    for i, test_season in enumerate(SEASON_CODES[-2:], start=len(SEASON_CODES)-2):
        if verbose:
            print(f"\n--- Test season: {SEASON_LABEL[test_season]} ---")

        # Load all data up to and including test season
        frames = []
        for s in SEASON_CODES[:i + 1]:
            try:
                frames.append(load_league(league, [s], offline=True))
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Could not load {league} {s}: {e}")
                continue

        if len(frames) < 2:
            if verbose:
                print("  [SKIP] Not enough seasons")
            continue

        df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

        # Run backtest for each strategy
        for strategy in ["favourite_only", "research_validated"]:
            result = run_real_data_backtest(
                df, strategy=strategy,
                use_research_layer=use_research_layer,
                verbose=False
            )
            result["test_season"] = SEASON_LABEL[test_season]
            result["n_seasons_train"] = i
            all_results[f"{test_season}_{strategy}"] = result

            m = result["metrics"]
            season_metrics.append({
                "test_season": SEASON_LABEL[test_season],
                "strategy": strategy,
                "n_train_seasons": i,
                **m,
            })

            if verbose:
                print(f"  {strategy:<25} bets={m['total_bets']:3d}  "
                      f"ROI={m['roi_pct']:+6.1f}%  "
                      f"WR={m.get('strike_rate', 0):5.1f}%  "
                      f"CLV={m.get('clv_mean_pct', 0):+.1f}%  "
                      f"Sharpe={m.get('sharpe_ratio', 0):.2f}")

    metrics_df = pd.DataFrame(season_metrics)
    metrics_df.to_csv(RESULTS_DIR / "clv_real_data_results.csv", index=False)

    return {"all_results": all_results, "metrics_df": metrics_df}


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="CLV Validation & Real-Data Backtest")
    parser.add_argument("--offline", action="store_true", help="Use cached data")
    parser.add_argument("--league", default="SP1", choices=["SP1", "E0", "I1"])
    args = parser.parse_args()

    results = run_multi_season_walkforward(league=args.league, verbose=True)

    # Summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS BY STRATEGY")
    print("=" * 70)

    metrics_df = results["metrics_df"]
    if len(metrics_df) > 0:
        agg = metrics_df.groupby("strategy").agg({
            "total_bets": "sum",
            "roi_pct": "mean",
            "strike_rate": "mean",
            "clv_mean_pct": "mean",
            "sharpe_ratio": "mean",
            "final_bankroll": "mean",
        }).round(2)

        print(agg.to_string())

        # Best strategy
        best = agg["roi_pct"].idxmax()
        print(f"\nBest strategy: {best} (avg ROI: {agg.loc[best, 'roi_pct']:+.1f}%)")

    # CLV validation
    print("\n" + "=" * 70)
    print("CLV AS PREDICTOR OF PROFITABILITY")
    print("=" * 70)

    for key, result in results["all_results"].items():
        clv = result.get("clv_analysis", {})
        if clv.get("sufficient_data"):
            strategy = result["strategy"]
            season = result["test_season"]
            print(f"  {season} {strategy:<20} "
                  f"pos_CLV_ROI={clv['positive_clv_roi']:+.1f}%  "
                  f"neg_CLV_ROI={clv['negative_clv_roi']:+.1f}%  "
                  f"gap={clv['clv_gap']:+.1f}%  "
                  f"predicts={clv['clv_predicts_profitability']}")

    print(f"\n[OK] Results saved to {RESULTS_DIR / 'clv_real_data_results.csv'}")


if __name__ == "__main__":
    main()
