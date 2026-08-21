#!/usr/bin/env python3
"""
Expand Data — download 13 leagues x 21 seasons from football-data.co.uk.

This gives us 95,000+ matches to properly test KDE, Bayesian, and
Mixture Monte Carlo models that need large samples to learn.

Usage:
    python data/expand_data.py
    python data/expand_data.py --offline  # use cached only
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REAL_DIR = PROJECT_ROOT / "data" / "real"

# All leagues available on football-data.co.uk
LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
    "B1": "Jupiler League",
    "SC0": "Scottish Premiership",
}

# Seasons: 2005/06 to 2025/26
# football-data.co.uk uses YY06 format: 0506 = 2005/06
SEASONS = []
SEASON_LABEL = {}
for y in range(5, 26):  # 05 to 25
    code = f"{y:02d}{(y+1)%100:02d}"  # e.g., 0506, 0607, ..., 2425
    SEASONS.append(code)
    SEASON_LABEL[code] = f"{2000+y}/{2000+y+1}"


def download_season(league: str, season: str, offline: bool = False) -> pd.DataFrame:
    """Download one season, normalizing to project schema."""
    cache = REAL_DIR / f"{league}_{season}.csv"

    if offline:
        if not cache.exists():
            return pd.DataFrame()
        raw = pd.read_csv(cache)
    else:
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
                raw = pd.read_csv(url)
                cache.write_bytes(raw.to_csv(index=False).encode())
            except Exception as e:
                print(f"  [WARN] {league} {season}: {e}")
                return pd.DataFrame()
        else:
            raw = pd.read_csv(cache)

    # Normalize columns
    rename_map = {
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
        "PSH": "pin_home", "PSD": "pin_draw", "PSA": "pin_away",
        "HS": "home_shots", "AS": "away_shots",
        "HST": "home_sot", "AST": "away_sot",
        "HC": "home_corners", "AC": "away_corners",
        "HY": "home_yellow", "AY": "away_yellow",
        "HR": "home_red", "AR": "away_red",
        "B365CH": "closing_odds_home", "B365CD": "closing_odds_draw",
        "B365CA": "closing_odds_away",
        "PSCH": "closing_odds_home_pin", "PSCD": "closing_odds_draw_pin",
        "PSCA": "closing_odds_away_pin",
        "MaxH": "best_odds_home", "MaxD": "best_odds_draw",
        "MaxA": "best_odds_away",
        "MaxCH": "best_closing_odds_home", "MaxCD": "best_closing_odds_draw",
        "MaxCA": "best_closing_odds_away",
        "HTHG": "ht_home_goals", "HTAG": "ht_away_goals",
    }

    df = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["league"] = LEAGUES.get(league, league)
    df["league_code"] = league
    df["season"] = SEASON_LABEL.get(season, season)

    # Keep essential columns
    essential = ["date", "home_team", "away_team", "home_goals", "away_goals",
                 "result", "league", "league_code", "season",
                 "odds_home", "odds_draw", "odds_away",
                 "pin_home", "pin_draw", "pin_away",
                 "closing_odds_home", "closing_odds_draw", "closing_odds_away",
                 "best_odds_home", "best_odds_draw", "best_odds_away",
                 "best_closing_odds_home", "best_closing_odds_draw", "best_closing_odds_away",
                 "home_shots", "away_shots", "home_sot", "away_sot",
                 "home_corners", "away_corners",
                 "home_yellow", "away_yellow",
                 "ht_home_goals", "ht_away_goals"]

    keep = [c for c in essential if c in df.columns]
    df = df[keep]
    df = df.dropna(subset=["home_goals", "away_goals", "result"])

    return df.sort_values("date").reset_index(drop=True)


def download_all(offline: bool = False, leagues: list = None):
    """Download all leagues and seasons."""
    target_leagues = leagues or list(LEAGUES.keys())

    print("=" * 70)
    print("EXPANDING DATASET — football-data.co.uk")
    print("=" * 70)
    print(f"Leagues: {len(target_leagues)}")
    print(f"Seasons: {len(SEASONS)} ({SEASONS[0]} to {SEASONS[-1]})")
    print(f"Est. total: ~95,000+ matches")
    print("=" * 70)

    all_frames = []
    total_matches = 0

    for league in target_leagues:
        league_name = LEAGUES.get(league, league)
        league_matches = 0

        for season in SEASONS:
            df = download_season(league, season, offline=offline)
            if len(df) > 0:
                all_frames.append(df)
                league_matches += len(df)
                total_matches += len(df)

        print(f"  {league_name:<25} {league_matches:5d} matches")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)

        # Save combined dataset
        output_path = REAL_DIR / "all_leagues_combined.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n{'='*70}")
        print(f"TOTAL: {total_matches} matches saved to {output_path}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Leagues: {combined['league'].nunique()}")
        print(f"Teams: {combined['home_team'].nunique()}")
        print(f"{'='*70}")

        return combined
    else:
        print("No data downloaded")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expand dataset")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--leagues", nargs="+", default=None,
                        help="Specific leagues to download")
    args = parser.parse_args()

    df = download_all(offline=args.offline, leagues=args.leagues)
