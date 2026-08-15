# Data Sources

The project supports **two data paths**:

## 1. Synthetic (default, fully reproducible)

`pipeline.generate_match_data()` builds a calibrated football world:

- 10 teams with latent strengths; goals ~ Poisson with home advantage.
- Bookmaker odds derived from the *true* probabilities + margin (~5–8%) +
  the favourite–longshot bias (`p_bookie ∝ p_true^0.88`).
- Independent closing odds (used for CLV).
- Seeded (`default_rng(42)`) — identical dataset on every machine.

This is the default so the project runs offline, instantly, and identically
for every reviewer. It is **not** claimed to be real match data.

## 2. Real data (optional, internet required)

`scripts/01_data_ingestion.py` downloads free, public data:

1. **football-data.co.uk** — historical results + bookmaker odds for the
   Premier League (E0), Bundesliga (D1), Serie A (I1), La Liga (SP1) and
   Ligue 1 (F1), seasons 2022–2024. Pinnacle closing odds (PSCH/PSCD/PSCA)
   are kept as the `closing_odds_*` columns so CLV is meaningful.
2. **openfootball / football.json** — World Cup 2022 fixtures & results (GitHub).

Other public sources that can be added later: football-data.org (free tier),
The Odds API (free tier), OpenLigaDB.

## How the pipeline picks a path

`pipeline.load_or_generate_data()` loads `data/processed/historical_matches.csv`
if it exists **and** has the expected schema (including closing odds);
otherwise it regenerates the synthetic dataset. So downloading real data
simply replaces the file and the rest of the pipeline is unchanged.
