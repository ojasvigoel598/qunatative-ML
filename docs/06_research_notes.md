# Research Notes — Ideas Used, Mapped to Implementation

Every externally-sourced idea that actually shaped this project is listed below
with its source, what we took, where it landed, and whether it improved
validation performance.

## Research → Implementation mapping

| ID | Source | Year | Main idea | What we adopted | Implementation | Improved validation? |
|----|--------|------|-----------|-----------------|----------------|----------------------|
| R1 | Dixon & Coles, *Modelling Association Football Scores and Inefficiencies in the Football Betting Market* (JRSS-C) | 1997 | Poisson goal model; full score-grid sum for P(H/D/A); market inefficiencies exist | Poisson regressions for home/away goals; probability from the full 0–8 score grid; the betting layer hunts residual inefficiency | `models/poisson_elo_model.py` (`predict`, `_poisson_pmf`) | Yes — baseline model, log-loss ≈ 0.96–0.98 vs 1.10 random baseline |
| R2 | Elo (1978) as adapted in club-football ratings literature | 1978+ | Dynamic team-strength ratings updated match by match | Sequential Elo with K=20; Elo difference feeds the Poisson regressions and every ML layer as features | `models/poisson_elo_model.py` (`_update_elo`, `prepare_features`) | Yes — Elo features are the strongest single input |
| R3 | Favorite–longshot bias (documented across betting markets, e.g. Harvard DASH work on MLB odds) | 2000s–2025 | Bookmakers overprice longshots / underprice favourites; bias strongest in opening odds, fades by closing | Synthetic bookie in `pipeline.py` shades favourite prices so genuine, stable value exists on favourites | `pipeline.py` (`generate_match_data`, `gamma` shading) | Yes — made edges ~10–15% and strike rate realistic instead of 777,820% ROI fantasy |
| R4 | Closing Line Value (CLV) literature — "beating the closing line predicts long-run profit better than win rate" | 2010s–2025 | The closing line is the sharpest estimate; track your price vs it | CLV is computed from independent closing odds in every backtest (`avg_clv_pct` ≈ 0 after fix) | `pipeline.py` (`compute_metrics`) | Yes — caught the `my_odds/my_odds - 1` bug (CLV was always 0.0) |
| R5 | Platt scaling / sigmoid calibration (Platt 1999); widely used for tree ensembles | 1999 | Raw ensemble probabilities are extreme; sigmoid map on out-of-fold predictions fixes it | Every tree model (GB, RF) is wrapped in `CalibratedClassifierCV(method="sigmoid")` | `models/ml_layer.py`, `scripts/04`, `scripts/05` | Yes — edge estimates became honest; ensemble behaves |
| R6 | Kelly criterion (Kelly 1956) + fractional-Kelly practice | 1956 | Stake = fraction of edge/odds; never full Kelly | Quarter-Kelly with a 5% stake cap; RL agent learns a Kelly multiplier instead of raw stakes | `pipeline.py`, `models/rl_staking_agent.py` | Yes — removed 4–5× overstaking that caused −76% drawdowns |
| R7 | Time-aware / walk-forward validation (standard in financial ML) | — | Never random-split temporal data; test = genuinely unseen future | Train/validation/test chronological split in the backtest; expanding-window season-by-season backtest on real data | `pipeline.py`, `scripts/05_season_backtest.py` | Yes — this is the headline validation |
| R8 | Leakage-free rolling features (shift before rolling) | — | A match's own goals must never be a feature for itself | `.shift(1)` on all rolling form; online feature builder in `scripts/05` | `models/ml_layer.py`, `scripts/05_season_backtest.py` | Yes — tests assert zero leakage |
| R9 | Academic consensus: 3-way football outcome accuracy caps ≈ 50–58% (aleatoric uncertainty dominates) | 2020s | Draws are near-unpredictable; accuracy alone is a weak target | We report calibration (ECE, Brier, log-loss) alongside accuracy and never claim 65%+ on real data | `scripts/05_season_backtest.py` (`evaluate`) | N/A — set honest expectations |
| R10 | Domain-transfer testing (train one league, test another) | — | Test generalisation, not just fit | Cross-league transfer experiments: synthetic→La Liga/EPL and La Liga↔EPL on real data | `scripts/04_deep_learning_transfer.py`, `scripts/05_season_backtest.py` | Informative — real-data cross-league transfer is positive but smaller than within-league |

## Not adopted (and why)

* **Dixon–Coles low-score dependence correction (τ)** — a genuine improvement
  (~1–3% log-likelihood) but the project's synthetic bookie already encodes a
  draw bias; adding τ would complicate the calibration story without changing
  the methodology conclusions. Listed as a future improvement.
* **xG / shots / advanced stats** — not available in the football-data.co.uk
  schema used for the real-data experiments; adding them is a data task, not a
  model task (see `docs/01_data_sources.md`).
* **Full Kelly / aggressive staking** — explicitly rejected by design.

## GitHub comparison

Compared with representative open-source football-prediction repositories
found in research (e.g. `bk1210/Football-Match-Outcome-Prediction`,
`ohad6k/football-match-prediction`, `Max-tech334` EPL prediction, typical
`predicting-football-results` tutorials).

| Aspect | Typical open-source repos | This project |
|--------|---------------------------|--------------|
| Architecture | Single notebook; pipeline embedded in cells | Modular `models/` + `pipeline.py` + `scripts/` + `demo/` + `tests/` |
| Validation | Often random train/test split | Chronological train/val/test + expanding-window season backtest + cross-league transfer |
| Calibration | Rarely handled | Sigmoid-calibrated probabilities + ECE/Brier/log-loss reported |
| Leakage checks | Usually absent | Automated leakage tests (`tests/test_pipeline.py`) |
| Betting layer | Sometimes a naive edge rule | Fractional-Kelly staking + Q-learning staking agent + $1M Monte-Carlo simulation with drawdown/ruin metrics |
| Baselines | Often majority-class only | Majority, market odds, PoissonElo, ridge, GB, RF, deep nets |
| Honesty of results | Frequently overfit/backtested on training data | Explicit in-sample vs out-of-sample labels; synthetic world disclosed as such |

**What they do better than us:** richer feature sets (shots, H2H, market odds as
features), more seasons of real data, production web scraping.

**What we do better:** honest time-aware methodology, calibration, leakage
tests, risk-managed staking simulation, deep-learning + transfer experiments,
and fully reproducible seeded runs.

**What we should adopt (future):** real odds as a feature, xG data, H2H stats,
more leagues/seasons.

**What we should not adopt:** random splits, tuning on the test set, or
"accuracy on the training set" as a headline result.
