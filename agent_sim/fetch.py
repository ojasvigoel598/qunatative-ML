#!/usr/bin/env python3
"""
On-demand data fetch for the multi-league agent simulation.

Fetches real seasons from the free public API at football-data.co.uk (no key,
no authentication) exactly when they are needed for a run, instead of
requiring large stored files.  A small in-memory cache avoids re-downloading
the same season twice in one process; with --offline it reads the cached
CSVs in data/real/ (the same files the rest of the repo uses).

Leagues (football-data.co.uk codes):
    SP1 La Liga · E0 Premier League · D1 Bundesliga · I1 Serie A

Usage:
    from agent_sim.fetch import fetch_seasons, LEAGUES, SEASON_LABEL
    df = fetch_seasons(["SP1", "E0"], ["2324", "2425"], offline=False)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = PROJECT_ROOT / "data" / "real"

LEAGUES = {"SP1": "La Liga", "E0": "Premier League",
           "D1": "Bundesliga", "I1": "Serie A"}

# season codes as they appear in the URL (2021 = 2020/21)
SEASON_CODES = ["2021", "2122", "2223", "2324", "2425", "2526"]
SEASON_LABEL = {"2021": "2020/21", "2122": "2021/22", "2223": "2022/23",
                "2324": "2023/24", "2425": "2024/25", "2526": "2025/26"}

# seasons considered COMPLETE (results fully available) — walk seasons are
# chosen only from these so results genuinely exist for every match.
COMPLETE_SEASONS = ["2122", "2223", "2324", "2425", "2526"]

# B365 = public line you can actually bet at; Pinnacle = sharp reference line
ODDS_MAP = {"B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
            "PSH": "pin_home", "PSD": "pin_draw", "PSA": "pin_away"}
KEEP = ["date", "home_team", "away_team", "home_goals", "away_goals",
        "result", "league", "season",
        "odds_home", "odds_draw", "odds_away",
        "pin_home", "pin_draw", "pin_away"]

_cache: dict = {}


def download_season(league: str, season: str) -> pd.DataFrame:
    """Download one real season from the public API, normalised to the schema."""
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    raw = pd.read_csv(url)
    raw = raw.rename(columns={"Date": "date", "HomeTeam": "home_team",
                              "AwayTeam": "away_team", "FTHG": "home_goals",
                              "FTAG": "away_goals", "FTR": "result",
                              **ODDS_MAP})
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y", errors="coerce")
    raw["league"] = LEAGUES[league]
    raw["season"] = SEASON_LABEL[season]
    keep = [c for c in KEEP if c in raw.columns]
    df = raw[keep].copy()
    df = df.dropna(subset=["date", "home_goals", "away_goals", "result"])
    # odds may be missing for a few matches -> NaN (agent records a no-bet)
    return df.sort_values("date").reset_index(drop=True)


def fetch_season(league: str, season: str, offline: bool = False) -> pd.DataFrame:
    """Get one league-season, downloading on demand (cached in memory)."""
    key = (league, season)
    if key in _cache:
        return _cache[key]
    cache_file = REAL_DIR / f"{league}_{season}.csv"
    if offline:
        if not cache_file.exists():
            sys.exit(f"[FAIL] --offline but {cache_file} is missing. "
                     f"Run once online to cache the data.")
        df = pd.read_csv(cache_file)
        # rebuild normalised columns from the cached raw CSV
        raw = df
        raw = raw.rename(columns={"Date": "date", "HomeTeam": "home_team",
                                  "AwayTeam": "away_team", "FTHG": "home_goals",
                                  "FTAG": "away_goals", "FTR": "result",
                                  **ODDS_MAP})
        raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y",
                                     errors="coerce")
        raw["league"] = LEAGUES[league]
        raw["season"] = SEASON_LABEL[season]
        keep = [c for c in KEEP if c in raw.columns]
        raw = raw[keep].dropna(subset=["date", "home_goals", "away_goals",
                                       "result"])
        df = raw.sort_values("date").reset_index(drop=True)
    else:
        df = download_season(league, season)
        # persist to the shared cache so later --offline runs work
        try:
            REAL_DIR.mkdir(parents=True, exist_ok=True)
            raw_url = pd.read_csv(
                f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv")
            cache_file.write_bytes(raw_url.to_csv(index=False).encode())
        except Exception:
            pass
    _cache[key] = df
    return df


def fetch_seasons(leagues, seasons, offline: bool = False) -> pd.DataFrame:
    """Concatenate several league-seasons into one chronological frame."""
    frames = [fetch_season(l, s, offline) for l in leagues for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cache real multi-league data")
    parser.add_argument("--leagues", nargs="+", default=["SP1", "E0", "D1", "I1"])
    parser.add_argument("--seasons", nargs="+", default=SEASON_CODES)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    for lg in args.leagues:
        for s in args.seasons:
            df = fetch_season(lg, s, args.offline)
            print(f"  {LEAGUES[lg]:<15} {SEASON_LABEL[s]:<9} "
                  f"{len(df):>4} matches")
