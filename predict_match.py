#!/usr/bin/env python3
"""
Match Prediction Interface (betting-user card).

Predicts a fixture from the command line and reports the full reasoning
chain: model probability -> uncertainty -> bookmaker probability -> best
odds -> fair odds -> edge -> EV -> bet/no-bet -> CLV (where available).

    python predict_match.py --home "Inter" --away "Juventus"
    python predict_match.py --home "Arsenal" --away "Chelsea" --league E0
    python predict_match.py --home "Real Madrid" --away "Barcelona" \
        --odds-home 1.95 --odds-draw 3.60 --odds-away 4.20

How it works
------------
1. Loads real historical data for the league (cached in data/real/; downloads
   the previous seasons on first use).  If the fixture was already played it
   uses the recorded B365/Pinnacle/Max odds; otherwise you can pass odds from
   your own bookmaker.
2. Trains the Poisson + Elo model (with the Dixon-Coles correction, which is
   appropriate for real football data) and the calibrated Gradient Boosting
   layer on that history, using ONLINE features only.
3. Predicts the fixture, blending both layers, and reports the structured
   answer plus the value analysis against the odds.

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
from analysis.match_analysis import analyze_match  # noqa: E402
from data.real_data import LEAGUES, get_season, load_league  # noqa: E402
from models.ml_layer import MLFootballPredictor  # noqa: E402
from models.poisson_elo_model import PoissonEloModel  # noqa: E402

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
TRAIN_SEASONS = SEASONS[:-1]  # predict the current season out-of-sample
BASE_HOME_GOALS, BASE_AWAY_GOALS, BASE_ELO = 1.6, 1.3, 1500.0


def rolling_form(df: pd.DataFrame, team: str, is_home: bool, window: int = 5) -> float:
    """Average goals for the team's last `window` matches in this role (past only)."""
    if is_home:
        sub = df[df["home_team"] == team]["home_goals"]
    else:
        sub = df[df["away_team"] == team]["away_goals"]
    return float(sub.tail(window).mean()) if len(sub) else (BASE_HOME_GOALS if is_home else BASE_AWAY_GOALS)


def find_fixture(df: pd.DataFrame, home: str, away: str):
    """Latest row where these two teams met (case-insensitive)."""
    h = df[df["home_team"].str.lower() == home.lower()]
    h = h[h["away_team"].str.lower() == away.lower()]
    if len(h):
        return h.iloc[-1]
    a = df[df["home_team"].str.lower() == away.lower()]
    a = a[a["away_team"].str.lower() == home.lower()]
    if len(a):
        row = a.iloc[-1].copy()
        # normalise: make the requested team the "home" side for display
        return row
    return None


def main():
    parser = argparse.ArgumentParser(description="Predict a football match")
    parser.add_argument("--home", required=True, help="home team")
    parser.add_argument("--away", required=True, help="away team")
    parser.add_argument("--league", default="SP1", choices=list(LEAGUES),
                        help="league code (SP1=La Liga, E0=Premier League, I1=Serie A)")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the synthetic world instead of real history")
    parser.add_argument("--odds-home", type=float, default=None,
                        help="decimal odds from your book (home win)")
    parser.add_argument("--odds-draw", type=float, default=None,
                        help="decimal odds from your book (draw)")
    parser.add_argument("--odds-away", type=float, default=None,
                        help="decimal odds from your book (away win)")
    parser.add_argument("--uncertainty-z", type=float, default=1.0,
                        help="require edge > z*uncertainty before betting (default 1.0; "
                             "0 disables the uncertainty guard)")
    args = parser.parse_args()

    if args.synthetic:
        df, _ = pipeline.load_or_generate_data(n_matches=1200, seed=42)
        data_source = "synthetic world (1,200 matches, seed 42)"
        # synthetic rows already carry both books + closing
        fixture = find_fixture(df, args.home, args.away)
        train_df = df
    else:
        train_df = load_league(args.league, seasons=TRAIN_SEASONS, offline=True)
        data_source = (f"real {LEAGUES[args.league]} history, "
                       f"{len(train_df):,} matches (cached in data/real/)")
        try:
            current = load_league(args.league, seasons=[SEASONS[-1]], offline=True)
        except SystemExit:
            current = pd.DataFrame()
        fixture = find_fixture(current, args.home, args.away)
        if fixture is None:
            fixture = find_fixture(train_df, args.home, args.away)

    if train_df.empty:
        sys.exit(f"[FAIL] No history available for {args.league}. "
                 f"Run once with internet, or pass --synthetic.")

    # ---- train layers (both use only past information; DC on: real data) --
    poisson = PoissonEloModel()
    poisson.train(train_df, verbose=False)
    ml = MLFootballPredictor(model_type="gradient_boosting")
    ml.train(poisson.prepare_features(train_df), verbose=False)

    # ---- per-team inputs (kept from the original interface) ---------------
    home_elo = poisson.elo_ratings.get(args.home, BASE_ELO)
    away_elo = poisson.elo_ratings.get(args.away, BASE_ELO)
    home_form = rolling_form(train_df, args.home, is_home=True)
    away_form = rolling_form(train_df, args.away, is_home=False)
    cold = (args.home not in poisson.elo_ratings or args.away not in poisson.elo_ratings)

    p_pois = poisson.predict(args.home, args.away)
    p_ml = ml.predict_proba(args.home, args.away, home_elo=home_elo, away_elo=away_elo)
    blend = {k: 0.5 * p_pois[k] + 0.5 * p_ml[k]
             for k in ["home_win", "draw", "away_win"]}

    # ---- build the row the value analysis can probe -----------------------
    if fixture is not None and any(c in fixture.index for c in
                                   ("odds_home_b365", "odds_home", "best_odds_home")):
        row = fixture
    else:
        row = pd.Series({
            "date": pd.Timestamp.today().date(), "home_team": args.home,
            "away_team": args.away, "result": "",
            "odds_home_b365": args.odds_home or np.nan,
            "odds_draw_b365": args.odds_draw or np.nan,
            "odds_away_b365": args.odds_away or np.nan,
            "odds_home_pin": args.odds_home or np.nan,
            "odds_draw_pin": args.odds_draw or np.nan,
            "odds_away_pin": args.odds_away or np.nan,
        })
        if not (args.odds_home and args.odds_draw and args.odds_away):
            sys.exit("[FAIL] This fixture has no recorded odds and you did not "
                     "pass --odds-home/--odds-draw/--odds-away. Edge analysis "
                     "needs a price to compare against.")

    card = analyze_match(row, poisson, ml, uncertainty_z=args.uncertainty_z,
                         n_samples=300, seed=0)
    outcome = max(blend, key=blend.get)
    label = {"home_win": f"{args.home} (home win)", "draw": "Draw",
             "away_win": f"{args.away} (away win)"}[outcome]
    conf = blend[outcome]
    margin = conf - max(v for k, v in blend.items() if k != outcome)
    risk = "High" if margin < 0.05 else ("Medium" if margin < 0.10 else "Low")

    # ---- output -----------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"  {args.home}  vs  {args.away}   [{LEAGUES[args.league]}]")
    print("=" * 64)
    print(f"  Home win : {blend['home_win']:6.1%}")
    print(f"  Draw     : {blend['draw']:6.1%}")
    print(f"  Away win : {blend['away_win']:6.1%}")
    print(f"\n  ML prediction : {label}")
    print(f"  Confidence    : {conf:.0%} (margin over next outcome {margin:.1%}) -> {risk} risk")
    print(f"  Elo diff      : {home_elo - away_elo:+.0f} (home {home_elo:.0f}, away {away_elo:.0f})")
    print(f"  Recent form   : {args.home} {home_form:.2f} gf/home | {args.away} {away_form:.2f} ga/away")
    print(f"  PoissonElo    : H {p_pois['home_win']:.1%} / D {p_pois['draw']:.1%} / A {p_pois['away_win']:.1%}"
          f"  (Dixon-Coles rho {poisson.rho:+.3f})")
    print(f"  GradientBoost : H {p_ml['home_win']:.1%} / D {p_ml['draw']:.1%} / A {p_ml['away_win']:.1%}")
    print(f"  Baselines     : league home-win rate "
          f"{np.mean(train_df['result'] == 'H'):.1%}, draw {np.mean(train_df['result'] == 'D'):.1%}")
    print(f"  Data source   : {data_source}")
    print(f"  Freshness     : latest match in history "
          f"{train_df['date'].max().date() if pd.notna(train_df['date'].max()) else 'n/a'}")
    if cold:
        print("  WARNING       : one or both teams are NOT in the history "
              "(cold start - using league-average baseline for them).")

    # ---- value analysis (betting-user card) -------------------------------
    print("\n" + "-" * 64)
    print("  VALUE ANALYSIS (transparent reasoning chain)")
    print("-" * 64)
    print(f"  Model prob      : H {card['p_model_home_win']:.1%} / "
          f"D {card['p_model_draw']:.1%} / A {card['p_model_away_win']:.1%}")
    print(f"  Uncertainty     : H +-{card['unc_home_win']:.1%} / "
          f"D +-{card['unc_draw']:.1%} / A +-{card['unc_away_win']:.1%} "
          f"(Poisson Monte-Carlo)")
    print(f"  Books probed    : {card['n_books']} (price shopping across available books)")
    print(f"  Bookie implied  : H {card['p_bookie_home_win']:.1%} / "
          f"D {card['p_bookie_draw']:.1%} / A {card['p_bookie_away_win']:.1%} "
          f"(margin removed)")
    print(f"  Best odds       : H {card['best_odds_home_win']:.2f} / "
          f"D {card['best_odds_draw']:.2f} / A {card['best_odds_away_win']:.2f} "
          f"(across {card['n_books']} books)")
    if card["best_outcome"]:
        print(f"  Fair odds       : model {card['fair_odds_model']:.2f} vs "
              f"bookie {card['fair_odds_bookie']:.2f} (best outcome "
              f"'{card['best_outcome']}')")
        print(f"  Edge            : {card['edge_pct']:+.2f}%  "
              f"(uncertainty +-{card['edge_uncertainty_pct']:.2f}%)")
        print(f"  EV per unit     : {card['ev_per_unit_pct']:+.2f}%")
        if card["clv_pct"] is not None:
            print(f"  CLV vs closing  : {card['clv_pct']:+.2f}%")
        print(f"  DECISION        : {card['decision']}"
              + (f"  ({card['reason']})" if card["reason"] else ""))
        if card["kelly_stake_frac"] > 0:
            print(f"  Suggested stake : {card['kelly_stake_frac']:.2%} of bankroll "
                  f"(quarter Kelly, capped)")
    print("=" * 64)
    print("  Honest note: this is research, not advice. Positive edge here means\n"
          "  the model disagrees with the price - it does NOT mean the model is\n"
          "  right. The project's own backtests show the model loses to a real\n"
          "  margin; CLV is the metric that tells you whether an edge is real.\n")


if __name__ == "__main__":
    main()
