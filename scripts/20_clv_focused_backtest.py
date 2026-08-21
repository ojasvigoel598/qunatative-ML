#!/usr/bin/env python3
"""
CLV-Focused Backtest — only bet when closing line confirms the edge.

This implements the highest-evidence strategy from the literature:
- Mints (2021): CLV is the strongest predictor of profitability
- Gramm & Owens (2006): Professional bettors achieve positive CLV

Strategy:
1. Model selects bet based on probability > bookmaker implied
2. Only place bet if closing line also shows value (CLV > 0)
3. Use fractional Kelly with strict caps
4. Test across multiple leagues and seasons

Expected outcome: Positive CLV should correlate with positive ROI.

Usage:
    python scripts/20_clv_focused_backtest.py --offline
"""

import argparse
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

RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Online Feature Builder (no leakage)
# ======================================================================
class OnlineFeatureBuilder:
    """Running Elo + rolling form; features use only past matches."""

    def __init__(self, elo_k=20.0, elo_base=1500.0):
        self.elo_k = elo_k
        self.elo_base = elo_base
        self.elo = defaultdict(lambda: elo_base)
        self.match_history = []

    def get_features(self, home_team, away_team):
        return {
            "home_elo": self.elo[home_team],
            "away_elo": self.elo[away_team],
            "elo_diff": self.elo[home_team] - self.elo[away_team],
        }

    def update(self, row):
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        home_r, away_r = self.elo[h], self.elo[a]
        exp_home = 1 / (1 + 10 ** ((away_r - home_r) / 400))
        actual = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        self.elo[h] += self.elo_k * (actual - exp_home)
        self.elo[a] += self.elo_k * ((1 - actual) - (1 - exp_home))

        self.match_history.append({
            "home_team": h, "away_team": a,
            "home_goals": hg, "away_goals": ag,
            "result": row["result"], "date": row.get("date"),
        })
        if len(self.match_history) > 30:
            self.match_history = self.match_history[-30:]

    def get_form(self, team, n=5):
        team_matches = [
            m for m in self.match_history
            if m["home_team"] == team or m["away_team"] == team
        ][-n:]
        if not team_matches:
            return {"form_pts": 1.33, "goals_scored": 1.5}
        pts = []
        for m in team_matches:
            if m["home_team"] == team:
                r = m["result"]
                pts.append(3 if r == "H" else (1 if r == "D" else 0))
            else:
                r = m["result"]
                pts.append(3 if r == "A" else (1 if r == "D" else 0))
        return {"form_pts": np.mean(pts), "goals_scored": np.mean([m["home_goals"] if m["home_team"] == team else m["away_goals"] for m in team_matches])}


# ======================================================================
# CLV-Focused Backtest
# ======================================================================
def run_clv_focused_backtest(df, league_name, test_season,
                              use_clv_filter=True,
                              clv_threshold=0.0,
                              odds_range=(1.6, 2.5),
                              edge_threshold=0.02,
                              kelly_fraction=0.20,
                              initial_bankroll=10000.0,
                              verbose=True):
    """Run CLV-focused backtest on real data.

    Args:
        df: Full dataset (train + test)
        league_name: League name for logging
        test_season: Season being tested
        use_clv_filter: Only bet if closing line confirms value
        clv_threshold: Minimum CLV to place bet (e.g., 0.0 = positive CLV)
        odds_range: (min_odds, max_odds) for bet selection
        edge_threshold: Minimum edge vs opening odds
        kelly_fraction: Fraction of Kelly to stake
        initial_bankroll: Starting bankroll
        verbose: Print results

    Returns:
        Dict with metrics and bets
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Split: 60% train, 40% test
    n = len(df)
    split_idx = int(n * 0.6)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if verbose:
        print(f"\n  {league_name} {test_season}: train={len(train_df)}, test={len(test_df)}")

    # Train on historical data
    builder = OnlineFeatureBuilder()

    # Build training features
    train_features = []
    for _, row in train_df.iterrows():
        feat = builder.get_features(row["home_team"], row["away_team"])
        train_features.append(feat)
        builder.update(row)

    train_feat_df = pd.DataFrame(train_features)
    train_feat_df["result"] = train_df["result"].values

    # Train ML model
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit

    target_map = {"H": 2, "D": 1, "A": 0}
    y_train = train_feat_df["result"].map(target_map)
    X_train = train_feat_df[["home_elo", "away_elo", "elo_diff"]].copy()

    ml = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                   learning_rate=0.05, min_samples_leaf=20,
                                   subsample=0.8, random_state=42),
        method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    )
    ml.fit(X_train, y_train)

    # Test walk-forward
    bets = []
    equity = [initial_bankroll]
    bankroll = initial_bankroll
    all_bets = []

    min_odds, max_odds = odds_range

    for _, row in test_df.iterrows():
        home, away = row["home_team"], row["away_team"]

        # Model prediction
        try:
            feat = builder.get_features(home, away)
            X = pd.DataFrame([feat])
            proba = ml.predict_proba(X)[0]
            probs = {"away_win": proba[0], "draw": proba[1], "home_win": proba[2]}
        except Exception:
            builder.update(row)
            continue

        # Bookmaker odds
        bookie = {}
        for side, col in [("home_win", "odds_home"), ("draw", "odds_draw"), ("away_win", "odds_away")]:
            if col in row.index and pd.notna(row[col]) and row[col] > 1.01:
                bookie[side] = row[col]

        if not bookie:
            builder.update(row)
            continue

        # Closing odds
        closing = {}
        for side, prefix in [("home_win", "closing_odds_home"), ("draw", "closing_odds_draw"), ("away_win", "closing_odds_away")]:
            if prefix in row.index and pd.notna(row[prefix]) and row[prefix] > 1.01:
                closing[side] = row[prefix]

        # Calculate edges
        edges = {}
        for side in probs:
            if side in bookie:
                edge = probs[side] * bookie[side] - 1.0
                edges[side] = edge

        # Find best edge
        best_side = max(edges, key=edges.get) if edges else None
        best_edge = edges.get(best_side, 0) if best_side else 0
        odds = bookie.get(best_side, 0)

        # Filter: odds range
        if not (min_odds <= odds <= max_odds):
            builder.update(row)
            continue

        # Filter: edge threshold
        if best_edge < edge_threshold:
            builder.update(row)
            continue

        # Filter: model probability
        model_prob = probs.get(best_side, 0)
        if model_prob < 0.40:
            builder.update(row)
            continue

        # CLV filter (key innovation)
        if use_clv_filter and best_side in closing:
            closing_odds = closing[best_side]
            clv = (closing_odds - odds) / odds

            if clv < clv_threshold:
                # Closing line moved against us → no value
                builder.update(row)
                continue

            # Use closing odds for more realistic edge calculation
            closing_edge = model_prob * closing_odds - 1.0
            if closing_edge < edge_threshold:
                builder.update(row)
                continue
        else:
            clv = 0.0
            closing_odds = odds

        # Kelly stake
        stake_frac = (best_edge / (odds - 1)) * kelly_fraction
        stake_frac = min(stake_frac, 0.08)  # max 8%
        stake = bankroll * stake_frac

        if stake < 50:
            builder.update(row)
            continue

        # Resolve bet
        win = row["result"].upper() == best_side[0].upper()
        profit = stake * (odds - 1) if win else -stake
        bankroll += profit

        bets.append({
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "match": f"{home} vs {away}",
            "market": best_side,
            "my_odds": odds,
            "closing_odds": closing_odds,
            "clv_pct": round(clv * 100, 2),
            "model_prob": round(model_prob, 4),
            "edge_pct": round(best_edge * 100, 2),
            "stake": round(stake, 2),
            "profit_loss": round(profit, 2),
            "bet_outcome": "Win" if win else "Lose",
            "running_bankroll": round(bankroll, 2),
        })
        equity.append(bankroll)

        builder.update(row)

    # Compute metrics
    bets_df = pd.DataFrame(bets)
    if len(bets_df) == 0:
        return {"metrics": {"total_bets": 0, "roi_pct": 0}, "bets_df": bets_df, "equity": equity}

    total_bets = len(bets_df)
    wins = int((bets_df["bet_outcome"] == "Win").sum())
    total_profit = float(bets_df["profit_loss"].sum())
    roi = total_profit / initial_bankroll * 100

    # Sharpe
    returns = np.diff(equity) / np.array(equity[:-1])
    bets_per_year = total_bets / max((df["date"].max() - df["date"].min()).days / 365.25, 0.1)
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bets_per_year)) if len(returns) > 1 and np.std(returns) > 0 else 0

    # Max drawdown
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    max_dd = float(abs(((equity_arr - peak) / peak).min()) * 100)

    # Profit factor
    gross_wins = float(bets_df.loc[bets_df["profit_loss"] > 0, "profit_loss"].sum())
    gross_losses = float(-bets_df.loc[bets_df["profit_loss"] < 0, "profit_loss"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # CLV analysis
    clv_arr = bets_df["clv_pct"].to_numpy()
    clv_mean = float(clv_arr.mean())
    clv_positive_rate = float(np.mean(clv_arr > 0) * 100)

    metrics = {
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
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else None,
        "final_bankroll": round(float(equity[-1]), 2),
        "clv_mean_pct": round(clv_mean, 2),
        "clv_positive_rate": round(clv_positive_rate, 2),
    }

    if verbose:
        print(f"    Bets: {total_bets}  Win Rate: {metrics['strike_rate']}%  "
              f"ROI: {roi:+.1f}%  CLV: {clv_mean:+.1f}%  Sharpe: {sharpe:.2f}")

    return {"metrics": metrics, "bets_df": bets_df, "equity": equity}


# ======================================================================
# Multi-League Testing
# ======================================================================
def run_multi_league_test(leagues=["SP1"],
                           test_seasons=["2526"],
                           verbose=True):
    """Test CLV-focused strategy across multiple leagues and seasons."""
    if verbose:
        print("=" * 70)
        print("CLV-FOCUSED BACKTEST — MULTI-LEAGUE")
        print("=" * 70)
        print("Strategy: Only bet when closing line confirms value")
        print("Filter: odds 1.6-2.5, edge >2%, CLV >0%")
        print("=" * 70)

    all_results = []

    for league in leagues:
        for test_season in test_seasons:
            try:
                # Load all data
                frames = []
                for s in SEASON_CODES:
                    try:
                        frames.append(load_league(league, [s], offline=True))
                    except Exception:
                        continue

                if len(frames) < 3:
                    continue

                # Use only last 3 seasons for speed
                df = pd.concat(frames[-3:], ignore_index=True).sort_values("date").reset_index(drop=True)

                # Run CLV-focused backtest
                result = run_clv_focused_backtest(
                    df, LEAGUES[league], SEASON_LABEL[test_season],
                    use_clv_filter=True,
                    clv_threshold=0.0,
                    odds_range=(1.6, 2.5),
                    edge_threshold=0.02,
                    kelly_fraction=0.20,
                    verbose=verbose,
                )

                result["league"] = LEAGUES[league]
                result["season"] = SEASON_LABEL[test_season]
                all_results.append(result)

            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {LEAGUES[league]} {test_season}: {e}")

    # Aggregate results
    if all_results:
        all_metrics = []
        for r in all_results:
            m = r["metrics"].copy()
            m["league"] = r["league"]
            m["season"] = r["season"]
            all_metrics.append(m)

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(RESULTS_DIR / "clv_focused_results.csv", index=False)

        if verbose:
            print("\n" + "=" * 70)
            print("AGGREGATE RESULTS BY LEAGUE")
            print("=" * 70)

            agg = metrics_df.groupby("league").agg({
                "total_bets": "sum",
                "roi_pct": "mean",
                "strike_rate": "mean",
                "clv_mean_pct": "mean",
                "sharpe_ratio": "mean",
                "final_bankroll": "mean",
            }).round(2)

            print(agg.to_string())

            # Overall
            total_bets = metrics_df["total_bets"].sum()
            avg_roi = metrics_df["roi_pct"].mean()
            avg_sharpe = metrics_df["sharpe_ratio"].mean()
            print(f"\n  Total bets: {total_bets}")
            print(f"  Average ROI: {avg_roi:+.1f}%")
            print(f"  Average Sharpe: {avg_sharpe:.2f}")

            # Best league
            best_league = agg["roi_pct"].idxmax()
            print(f"\n  Best league: {best_league} (ROI: {agg.loc[best_league, 'roi_pct']:+.1f}%)")

    return all_results


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="CLV-Focused Backtest")
    parser.add_argument("--offline", action="store_true", help="Use cached data")
    parser.add_argument("--leagues", nargs="+", default=["SP1", "E0", "I1"])
    args = parser.parse_args()

    results = run_multi_league_test(leagues=args.leagues, verbose=True)

    print("\n[OK] Results saved to backtests/results/clv_focused_results.csv")


if __name__ == "__main__":
    main()
