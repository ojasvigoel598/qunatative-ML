#!/usr/bin/env python3
"""
Match Prediction Interface.

Predicts a fixture from the command line, e.g.:

    python predict_match.py --home "Real Madrid" --away "Barcelona"
    python predict_match.py --home "Arsenal" --away "Chelsea" --league E0

How it works
------------
1. Loads real historical data for the league (cached in data/real/; downloads
   the five previous seasons on first use).  If no real data is available and
   --synthetic is passed, falls back to the synthetic world.
2. Trains the project's Poisson + Elo model and the calibrated Gradient
   Boosting layer on that history, using ONLINE features (a team's features
   only use matches before the prediction point).
3. Predicts the fixture, blending both layers, and reports the structured
   answer: probabilities, chosen outcome, Elo difference, recent form,
   baseline probabilities, confidence, risk and data freshness.

The CLI never needs internet at prediction time once data is cached.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402
from models.ml_layer import MLFootballPredictor  # noqa: E402
from models.poisson_elo_model import PoissonEloModel  # noqa: E402

REAL_DIR = PROJECT_ROOT / "data" / "real"
SEASONS = ["2122", "2223", "2324", "2425", "2526"]
LEAGUES = {"SP1": "La Liga", "E0": "Premier League"}
BASE_HOME_GOALS, BASE_AWAY_GOALS, BASE_ELO = 1.6, 1.3, 1500.0


def load_real_history(league: str) -> pd.DataFrame:
    """Concatenate the five previous seasons for a league (cached)."""
    parts = []
    for season in SEASONS[:-1]:
        cache = REAL_DIR / f"{league}_{season}.csv"
        if not cache.exists():
            REAL_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pd.read_csv(
                f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            ).to_csv(index=False).encode())
        df = pd.read_csv(cache)
        df = df.rename(columns={
            "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
            "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        })
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        parts.append(df[["date", "home_team", "away_team", "home_goals",
                         "away_goals", "result"]])
    return pd.concat(parts, ignore_index=True).dropna(
        subset=["home_goals", "away_goals", "result"]).reset_index(drop=True)


def rolling_form(df: pd.DataFrame, team: str, is_home: bool, window: int = 5) -> float:
    """Average goals for the team's last `window` matches in this role (past only)."""
    if is_home:
        sub = df[df["home_team"] == team]["home_goals"]
    else:
        sub = df[df["away_team"] == team]["away_goals"]
    return float(sub.tail(window).mean()) if len(sub) else (BASE_HOME_GOALS if is_home else BASE_AWAY_GOALS)


def main():
    parser = argparse.ArgumentParser(description="Predict a football match")
    parser.add_argument("--home", required=True, help="home team")
    parser.add_argument("--away", required=True, help="away team")
    parser.add_argument("--league", default="SP1", choices=list(LEAGUES),
                        help="league code (SP1=La Liga, E0=Premier League)")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the synthetic world instead of real history")
    args = parser.parse_args()

    if args.synthetic:
        df, _ = pipeline.load_or_generate_data(n_matches=1200, seed=42)
        data_source = "synthetic world (1,200 matches, seed 42)"
    else:
        df = load_real_history(args.league)
        data_source = f"real {LEAGUES[args.league]} history, 2021/22-2024/25 " \
                      f"({len(df):,} matches, cached in data/real/)"

    if df.empty:
        sys.exit(f"[FAIL] No history available for {args.league}. "
                 f"Run once with internet, or pass --synthetic.")

    # ---- train layers (both use only past information)
    poisson = PoissonEloModel()
    poisson.train(df)
    ml = MLFootballPredictor(model_type="gradient_boosting")
    ml.train(poisson.prepare_features(df), verbose=False)

    # ---- per-team inputs
    home_elo = poisson.elo_ratings.get(args.home, BASE_ELO)
    away_elo = poisson.elo_ratings.get(args.away, BASE_ELO)
    home_form = rolling_form(df, args.home, is_home=True)
    away_form = rolling_form(df, args.away, is_home=False)
    cold = (args.home not in poisson.elo_ratings or args.away not in poisson.elo_ratings)

    p_pois = poisson.predict(args.home, args.away)
    p_ml = ml.predict_proba(args.home, args.away, home_elo=home_elo, away_elo=away_elo)
    blend = {k: 0.5 * p_pois[k] + 0.5 * p_ml[k]
             for k in ["home_win", "draw", "away_win"]}

    # ---- output
    outcome = max(blend, key=blend.get)
    label = {"home_win": f"{args.home} (home win)", "draw": "Draw",
             "away_win": f"{args.away} (away win)"}[outcome]
    conf = blend[outcome]
    margin = conf - max(v for k, v in blend.items() if k != outcome)
    risk = "High" if margin < 0.05 else ("Medium" if margin < 0.10 else "Low")

    print("\n" + "=" * 60)
    print(f"  {args.home}  vs  {args.away}   [{LEAGUES[args.league]}]")
    print("=" * 60)
    print(f"  Home win : {blend['home_win']:6.1%}")
    print(f"  Draw     : {blend['draw']:6.1%}")
    print(f"  Away win : {blend['away_win']:6.1%}")
    print(f"\n  ML prediction : {label}")
    print(f"  Confidence    : {conf:.0%} (margin over next outcome {margin:.1%}) -> {risk} risk")
    print(f"  Elo diff      : {home_elo - away_elo:+.0f} (home {home_elo:.0f}, away {away_elo:.0f})")
    print(f"  Recent form   : {args.home} {home_form:.2f} gf/home | {args.away} {away_form:.2f} ga/away")
    print(f"  PoissonElo    : H {p_pois['home_win']:.1%} / D {p_pois['draw']:.1%} / A {p_pois['away_win']:.1%}")
    print(f"  GradientBoost : H {p_ml['home_win']:.1%} / D {p_ml['draw']:.1%} / A {p_ml['away_win']:.1%}")
    print(f"  Baselines     : league home-win rate {np.mean(df['result'] == 'H'):.1%}, "
          f"draw {np.mean(df['result'] == 'D'):.1%}")
    print(f"  Data source   : {data_source}")
    print(f"  Freshness     : latest match in history "
          f"{df['date'].max().date() if pd.notna(df['date'].max()) else 'n/a'}")
    if cold:
        print("  WARNING       : one or both teams are NOT in the history "
              "(cold start - using league-average baseline for them).")
    print("=" * 60)
    print("  Odds hint: fair odds ~ 1/p. Model edge vs a real bookmaker line = "
          "p * odds - 1. Bet only what you can afford to lose; this is research, not advice.\n")


if __name__ == "__main__":
    main()
