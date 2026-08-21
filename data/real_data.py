#!/usr/bin/env python3
"""
Shared real-data loader for multiple leagues (football-data.co.uk).

Covers La Liga (SP1), Premier League (E0) and Serie A (I1) for the seasons
2020/21 .. 2025/26.  Each season is normalised to the project schema with both
B365 and Pinnacle odds, cached to data/real/<LEAGUE>_<season>.csv, and can be
loaded offline (cached) or re-downloaded.

Usage:
    from data.real_data import get_season, load_league, LEAGUES

    df = get_season("I1", "2526")            # Serie A 2025/26 (downloads if needed)
    all_serie_a = load_league("I1", offline=False)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = PROJECT_ROOT / "data" / "real"

# football-data.co.uk league codes
LEAGUES = {"SP1": "La Liga", "E0": "Premier League", "I1": "Serie A"}

# season codes as they appear in the URL (2021 = 2020/21)
SEASON_CODES = ["2021", "2122", "2223", "2324", "2425", "2526"]
SEASON_LABEL = {"2021": "2020/21", "2122": "2021/22", "2223": "2022/23",
                "2324": "2023/24", "2425": "2024/25", "2526": "2025/26"}

ODDS_COLS = ["odds_home", "odds_draw", "odds_away",
             "pin_home", "pin_draw", "pin_away"]

# Closing lines and best-available prices (price shopping + CLV).  Named to
# match the synthetic world's schema so one accessor works for both.
CLOSING_COLS = ["closing_odds_home", "closing_odds_draw", "closing_odds_away",
                "closing_odds_home_pin", "closing_odds_draw_pin", "closing_odds_away_pin",
                "best_odds_home", "best_odds_draw", "best_odds_away",
                "best_closing_odds_home", "best_closing_odds_draw",
                "best_closing_odds_away"]

_CLOSING_RENAME = {
    "B365CH": "closing_odds_home", "B365CD": "closing_odds_draw", "B365CA": "closing_odds_away",
    "PSCH": "closing_odds_home_pin", "PSCD": "closing_odds_draw_pin", "PSCA": "closing_odds_away_pin",
    "MaxH": "best_odds_home", "MaxD": "best_odds_draw", "MaxA": "best_odds_away",
    "MaxCH": "best_closing_odds_home", "MaxCD": "best_closing_odds_draw",
    "MaxCA": "best_closing_odds_away",
}


def download_season(league: str, season: str, raw: pd.DataFrame = None) -> pd.DataFrame:
    """Normalise one real season to the project schema (with odds).

    `raw` is the raw football-data.co.uk frame; when omitted it is downloaded
    from the public API.  Callers with a local cache pass the cached raw frame
    so offline mode never touches the network.

    Keeps the closing lines (B365/Pinnacle) and the Max opening/closing
    columns so real data supports the same price-shopping and CLV analysis as
    the synthetic world.
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    df = raw if raw is not None else pd.read_csv(url)
    df = df.rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
        "PSH": "pin_home", "PSD": "pin_draw", "PSA": "pin_away",
        # Rich match statistics
        "HS": "home_shots", "AS": "away_shots",
        "HST": "home_shots_on_target", "AST": "away_shots_on_target",
        "HF": "home_fouls", "AF": "away_fouls",
        "HC": "home_corners", "AC": "away_corners",
        "HY": "home_yellow_cards", "AY": "away_yellow_cards",
        "HR": "home_red_cards", "AR": "away_red_cards",
        "HTHG": "home_goals_ht", "HTAG": "away_goals_ht",
        "HTR": "result_ht",
        **_CLOSING_RENAME})
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["league"] = LEAGUES[league]
    df["season"] = SEASON_LABEL[season]
    # Rich feature columns
    RICH_COLS = [
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_fouls", "away_fouls",
        "home_corners", "away_corners",
        "home_yellow_cards", "away_yellow_cards",
        "home_red_cards", "away_red_cards",
        "home_goals_ht", "away_goals_ht",
        "result_ht",
    ]
    keep = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "result", "league", "season"] + ODDS_COLS + CLOSING_COLS + RICH_COLS
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["home_goals", "away_goals", "result"])
    return df.sort_values("date").reset_index(drop=True)


def get_season(league: str, season: str, offline: bool = False) -> pd.DataFrame:
    """Load one season, downloading + caching on first use unless offline.

    Offline mode reads the cached raw CSV and normalises it in memory - it
    never touches the network.  (The old implementation called
    ``download_season`` unconditionally, which re-downloaded from the public
    API on EVERY call even with a populated cache, making ``offline=True``
    silently depend on the internet.)
    """
    cache = REAL_DIR / f"{league}_{season}.csv"
    if offline:
        if not cache.exists():
            sys.exit(f"[FAIL] --offline but {cache} missing. Run once without "
                     f"--offline to download.")
    else:
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            raw = pd.read_csv(
                f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv")
            cache.write_bytes(raw.to_csv(index=False).encode())
    return download_season(league, season, raw=pd.read_csv(cache))


def load_league(league: str, seasons=None, offline: bool = True) -> pd.DataFrame:
    """Concatenate all requested seasons of a league, sorted by date."""
    seasons = seasons or SEASON_CODES
    frames = [get_season(league, s, offline) for s in seasons]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


RICH_COLS = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY"]


def get_season_rich(league: str, season: str, offline: bool = True) -> pd.DataFrame:
    """Like `get_season` but keeps rich per-match columns (shots, corners,
    cards) and the raw B365 columns, for the sequence models."""
    cache = REAL_DIR / f"{league}_{season}.csv"
    if offline and not cache.exists():
        get_season(league, season, offline=False)
    df = pd.read_csv(cache)
    df = df.rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away"})
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["league"] = LEAGUES[league]
    df["season"] = SEASON_LABEL[season]
    keep = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "result", "league", "season", "odds_home", "odds_draw",
            "odds_away"] + RICH_COLS
    for c in ["odds_home", "odds_draw", "odds_away"]:
        if c not in df.columns:
            df[c] = 1.0 / 3.0
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["home_goals", "away_goals", "result"])
    return df.sort_values("date").reset_index(drop=True)


def load_league_rich(league: str, seasons=None, offline: bool = True) -> pd.DataFrame:
    """Concatenate rich seasons of a league, sorted by date."""
    seasons = seasons or SEASON_CODES
    frames = [get_season_rich(league, s, offline) for s in seasons]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download real league data")
    parser.add_argument("--league", choices=list(LEAGUES), default="I1")
    args = parser.parse_args()
    for s in SEASON_CODES:
        df = get_season(args.league, s)
        print(f"  {LEAGUES[args.league]} {SEASON_LABEL[s]}: {len(df)} matches "
              f"(cached {REAL_DIR / f'{args.league}_{s}.csv'})")
