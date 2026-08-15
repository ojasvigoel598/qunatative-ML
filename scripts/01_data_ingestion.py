#!/usr/bin/env python3
"""
Data Ingestion for the Quantitative Sports Betting Model.

Two data paths:

1. **REAL data** (default): downloads free historical results + odds from
   football-data.co.uk (EPL, Bundesliga, Serie A, La Liga, Ligue 1, 2022-2024)
   and the openfootball World Cup dataset.  Closing odds (Pinnacle) are kept
   where available so CLV can be computed.
2. **SYNTHETIC fallback**: if the download is unavailable or you just want a
   reproducible demo, use ``--synthetic`` (or simply run ``run_full_ml_rl.py``,
   which auto-generates the dataset).

Usage:
    python scripts/01_data_ingestion.py          # real data (internet needed)
    python scripts/01_data_ingestion.py --synthetic
"""

import argparse
import json

import pandas as pd
import requests
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = {"E0": "Premier League", "D1": "Bundesliga", "I1": "Serie A",
           "SP1": "La Liga", "F1": "Ligue 1"}
SEASONS = ["2022", "2023", "2024"]


def download_football_data_co_uk():
    base_url = "https://www.football-data.co.uk/mmz4281/"
    print("Downloading from football-data.co.uk...")
    for season in SEASONS:
        for league_code in LEAGUES:
            url = f"{base_url}{season}/{league_code}.csv"
            filename = f"{league_code}_{season}.csv"
            filepath = RAW_DIR / filename
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(r.content)
                    print(f"  [OK] Downloaded {filename}")
            except Exception as e:
                print(f"  [FAIL] {filename}: {e}")
    print("football-data.co.uk downloads complete.")


def download_openfootball_wc():
    print("Downloading openfootball WC data...")
    wc_url = "https://raw.githubusercontent.com/openfootball/football.json/master/2022/fifa-world-cup.json"
    try:
        r = requests.get(wc_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            filepath = RAW_DIR / "world_cup_2022.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  [OK] Downloaded {filepath.name}")
    except Exception as e:
        print(f"  [FAIL] WC download: {e}")


def process_historical_data():
    """Combine raw CSVs into data/processed/historical_matches.csv.

    Uses Pinnacle closing odds (PSCH/PSCD/PSCA) as closing_odds_* columns so
    the pipeline's CLV metric is meaningful; falls back to the B365 opening
    price when Pinnacle is missing.
    """
    print("Processing historical match data...")
    all_matches = []
    for csv_file in RAW_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            df = df.rename(columns={
                "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
                "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
                "B365H": "odds_home_b365", "B365D": "odds_draw_b365", "B365A": "odds_away_b365",
                "PSCH": "closing_odds_home", "PSCD": "closing_odds_draw", "PSCA": "closing_odds_away",
            })
            league = csv_file.stem.split("_")[0]
            season = csv_file.stem.split("_")[1]
            df["league"] = league
            df["season"] = season
            df["source"] = "football-data.co.uk"
            keep_cols = ["date", "home_team", "away_team", "home_goals", "away_goals",
                         "result", "league", "season", "source",
                         "odds_home_b365", "odds_draw_b365", "odds_away_b365",
                         "closing_odds_home", "closing_odds_draw", "closing_odds_away"]
            df = df[[c for c in keep_cols if c in df.columns]]
            # Where Pinnacle closing odds are missing, use the B365 opening price.
            for col, open_col in [("closing_odds_home", "odds_home_b365"),
                                  ("closing_odds_draw", "odds_draw_b365"),
                                  ("closing_odds_away", "odds_away_b365")]:
                if col not in df.columns:
                    df[col] = df[open_col]
                else:
                    df[col] = df[col].fillna(df[open_col])
            all_matches.append(df)
            print(f"  [OK] Processed {csv_file.name} ({len(df)} matches)")
        except Exception as e:
            print(f"  [FAIL] Error processing {csv_file.name}: {e}")

    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], format="%d/%m/%Y", errors="coerce")
        combined = combined.dropna(subset=["date", "home_goals", "away_goals"])
        combined.to_csv(PROCESSED_DIR / "historical_matches.csv", index=False)
        print(f"[OK] Saved combined historical data: {len(combined)} matches")
    else:
        print("[WARN] No raw files found - run the download step first.")


def main():
    parser = argparse.ArgumentParser(description="Sports betting data ingestion")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip downloads; generate the reproducible synthetic dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("SPORTS BETTING MODEL - DATA INGESTION")
    print("=" * 60)

    if args.synthetic:
        import pipeline
        pipeline.load_or_generate_data(regenerate=True)
    else:
        download_football_data_co_uk()
        download_openfootball_wc()
        process_historical_data()
        print("\nIngestion complete. Closing odds extracted for CLV.")

    print("\nNext: python run_full_ml_rl.py  (train + backtest the full pipeline)")


if __name__ == "__main__":
    main()
