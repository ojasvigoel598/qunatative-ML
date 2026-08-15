#!/usr/bin/env python3
"""
Real ATP tennis data with odds — fetched on demand from the free public
archive at tennis-data.co.uk (no key, no login).  The site serves yearly
Excel workbooks (https://www.tennis-data.co.uk/YYYY/YYYY.xlsx); we normalise
them to a compact CSV schema shared by the tennis walk-forward.

This is a DIFFERENT SPORT from football: 2-outcome matches (no draw), a
surface structure (Hard/Clay/Grass/Carpet), and a different data granularity
(winner/loser + ranks instead of home/away + goals).

Schema (data/tennis/atp_YYYY.csv):
    date, tournament, round, surface, winner, loser,
    wrank, lrank,                  # ATP rank at match time (known pre-match)
    odds_winner, odds_loser,       # B365 (public line you can actually bet)
    pin_winner, pin_loser,         # Pinnacle (sharp reference for CLV)
    max_winner, max_loser          # best price across bookmakers
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENNIS_DIR = PROJECT_ROOT / "data" / "tennis"

# public / sharp / max odds columns as they appear in the xlsx
ODDS_COLS = {"B365W": "odds_winner", "B365L": "odds_loser",
             "PSW": "pin_winner", "PSL": "pin_loser",
             "MaxW": "max_winner", "MaxL": "max_loser"}
KEEP = ["date", "tournament", "round", "surface", "winner", "loser",
        "wrank", "lrank",
        "odds_winner", "odds_loser", "pin_winner", "pin_loser",
        "max_winner", "max_loser"]

_cache: dict = {}


def _download_xlsx(year: int, retries: int = 4, timeout: int = 180) -> bytes:
    """Download one yearly workbook over plain http (https is TLS-broken on
    this host); retries because the host is occasionally slow."""
    import urllib.request

    url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:                     # noqa: BLE001
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"tennis-data.co.uk download failed for {year}: {last}")


def fetch_season(year: int, refresh: bool = False) -> pd.DataFrame:
    """Get one ATP season, normalised to the shared schema (cached as CSV)."""
    if year in _cache and not refresh:
        return _cache[year]
    TENNIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TENNIS_DIR / f"atp_{year}.csv"
    xlsx_path = TENNIS_DIR / f"atp_{year}.xlsx"
    if csv_path.exists() and not refresh:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        _cache[year] = df
        return df

    if not xlsx_path.exists() or refresh:
        data = _download_xlsx(year)
        xlsx_path.write_bytes(data)

    raw = pd.read_excel(xlsx_path)
    raw = raw.rename(columns={"Date": "date", "Tournament": "tournament",
                              "Round": "round", "Surface": "surface",
                              "Winner": "winner", "Loser": "loser",
                              "WRank": "wrank", "LRank": "lrank",
                              **ODDS_COLS})
    keep = [c for c in KEEP if c in raw.columns]
    df = raw[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "winner", "loser", "surface"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    _cache[year] = df
    return df


def fetch_seasons(years) -> pd.DataFrame:
    """Concatenate several seasons into one chronological frame."""
    frames = [fetch_season(int(y)) for y in years]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Cache real ATP tennis data")
    p.add_argument("--years", nargs="+", type=int,
                   default=list(range(2016, 2026)))
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    for y in args.years:
        df = fetch_season(y, refresh=args.refresh)
        print(f"  ATP {y}: {len(df):>5} matches, "
              f"{df['odds_winner'].notna().sum():>5} with B365 odds")
    sys.exit(0)
