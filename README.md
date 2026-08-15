<div align="center">

# ⚽ Quantitative Sports Betting Model

**A complete, reproducible machine-learning pipeline that prices football matches, finds value bets, and backtests a staking strategy — Poisson + Elo, Gradient Boosting, and Q-Learning.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/)
[![Tests](https://img.shields.io/badge/tests-12%20passing-brightgreen.svg)](#validation)

![Pipeline architecture](assets/architecture.png)

</div>

---

> ## 🚧 Project status: ongoing work-in-progress
>
> This repository is a **living research project** — it is published as an honest, runnable snapshot of work in progress, not a finished product. The distinctions below are maintained as the project evolves:
>
> - ✅ **Working / current** — core pipeline (PoissonElo + ML + RL), prediction CLI, real-data season backtests, walk-forward agent simulation, $1M simulations, tests, explained notebook.
> - 🧪 **Experimental** — deep-learning transfer (PyTorch/TF), adaptive cross-league/cross-sport transfer, LSTM/GRU state-space models, the dynamic thinking layer, hidden-signal CLV studies. These run, but results are reported honestly (often *not* beating the market) and may change.
> - 🗺️ **Planned** — see [Future improvements](#future-improvements).
> - ⚠️ **Known limitations** — see [Limitations](#limitations). Nothing here is financial or gambling advice; the default dataset is synthetic.
>
> Every commit is a small, tested, self-contained contribution, so the repository is always in a usable state.

---

## Overview

**The problem.** Beating a sportsbook is hard: bookmakers set prices from the same public information you have, then add a margin on top. Winning consistently requires a model that is *genuinely better* at estimating match probabilities — and even then, edge detection is fragile, staking is risky, and a single backtest path tells you almost nothing.

**What was built.** A three-layer quantitative betting system:

1. **Poisson + Elo hybrid** — Elo ratings feed two Poisson regressions that produce `P(home win)`, `P(draw)`, `P(away win)` for any fixture.
2. **Calibrated Gradient-Boosting ML layer** — a team-aware classifier on the same features, sigmoid-calibrated because `edge = p × odds − 1` is brutally sensitive to miscalibration.
3. **Q-Learning staking agent** — learns *how aggressively to bet* (Kelly multipliers) from realized bets on a validation split.

All three are glued together by a shared engine (`pipeline.py`) that runs a chronological **train / validation / test** backtest with honest metrics: ROI, strike rate, Sharpe, max drawdown, profit factor, and **CLV** (closing line value) computed from independent closing odds.

**Why it matters (portfolio-wise).** This is an end-to-end ML project with the rigor that separates a "notebook demo" from a defensible system: no data leakage, calibrated probabilities, a held-out test split that is never tuned on, a Monte-Carlo simulation instead of one lucky path, and a 12-test verification suite.

> **Honesty first.** The default data is a *synthetic, calibrated world* — the bookmaker odds are generated from the same true probabilities the model learns, plus a margin and the well-documented favourite–longshot bias. The backtest validates the **methodology**; it is **not** a prediction of real-world returns. See [Limitations](#limitations).

---

## Key features

- **Poisson + Elo hybrid** — dynamic team strength + count-model goal expectations.
- **Calibrated ML layer** — Gradient Boosting with sigmoid calibration and team-aware inference.
- **RL staking** — tabular Q-learning over Kelly multipliers (fixes the over-staking of the original design).
- **Deep learning** — a **PyTorch MLP** and a **TensorFlow hybrid** (MLP fused with PoissonElo outputs), trained on all data and tested for cross-league transfer.
- **Adaptive / dynamic model** — a league-agnostic model (`models/adaptive_model.py`) that keeps learning online: Elo + rolling form update after every match, and the ML layer refits on a rolling window when scheduled or when its rolling Brier drifts. Trained on **real Serie A**, it transfers to unseen Serie A 2025/26, La Liga, the Premier League and even a different sport (`scripts/10_adaptive_transfer.py`).
- **State-space models** — an **LSTM/GRU** over each team's rolling match sequence (`models/lstm_model.py`) models changing form directly; tested head-to-head against every baseline under the point-in-time protocol, plus a database-vs-model experiment (`scripts/11_lstm_state_test.py`).
- **Dynamic thinking layer** — `models/dynamic_thinking.py` makes every decision adaptive: it fuses the trained model with the **public-vs-sharp market split** (a hidden signal), **re-weights model-vs-market from its own rolling Brier**, shrinks stakes by uncertainty and drawdown, and switches to survival mode below 10% of the bankroll — now **confidence-aware**: its confidence (how far the top outcome sits above the uniform ⅓) modulates the calibration blend, gates immediate base-model refits when confidence decays, and scales stakes up only when it is genuinely more sure. Wired into the $1M simulation (`demo/simulation_dynamic.py`) and its live video.
- **Real-data, time-aware validation** — expanding-window season-by-season backtest on five real **La Liga + Serie A + Premier League** seasons plus cross-league transfer tests.
- **Rendered demo videos** — real `.mp4` walkthroughs of the $1M simulation and a point-in-time **Serie A 2025/26 replay**, rendered from actual run outputs (no fabricated footage).
- **Prediction CLI** — `predict_match.py --home "Real Madrid" --away "Barcelona"` prints probabilities, Elo difference, form, baselines, confidence and risk.
- **No leakage** — chronological split, Elo updated sequentially, rolling form features shifted by one match.
- **Real CLV** — computed from independent closing odds (the original code computed `odds/odds − 1 = 0`; fixed).
- **Reproducible** — every random draw is seeded; running anything twice gives identical results.
- **Monte-Carlo $1M simulation** — answers *"if I invested $1,000,000, what happens?"* with a distribution, not a single number.
- **Explained notebook** — `notebooks/01_explained_ml_pipeline.ipynb` walks every step end-to-end (22 sections, fully executed): the core PoissonElo + ML + RL pipeline, deep learning (PyTorch/TF), real-data experiments, the adaptive model, the confidence-aware dynamic thinking layer, the hidden-signals CLV study, the leak-free multi-league walk-forward agent, LSTM/GRU, and the $1M simulations.
- **Verification suite** — 12 tests covering leakage, calibration, CLV, reproducibility.

---

## Architecture / workflow

```
Input (synthetic world or football-data.co.uk CSV)
  ↓
Preprocessing: Elo features · shifted rolling form
  ↓
Train / validation / test split (65 / 15 / 20, chronological)
  ↓
PoissonElo model ──┐
                   ├─► Hybrid ensemble  p = (p_poisson + p_ml) / 2
ML (Gradient Boosting) ──┘
  ↓
Value detection:  edge = p × odds − 1 > 3%  and  p ≥ 0.40
  ↓
Q-Learning staking agent (trained on validation bets)
  ↓
Backtest: resolve bets · track bankroll · record CLV
  ↓
Metrics & visualisations: ROI · Sharpe · max DD · equity curve · CLV
```

See [`docs/02_model_architecture.md`](docs/02_model_architecture.md) for the detailed description of each layer.

---

## Results

All numbers below are from the **canonical run** (`seed=42`, 1,200 synthetic matches) and are reproduced by running the commands in [Usage](#usage).

### Backtest — two configurations

| Metric | PoissonElo + Kelly | PoissonElo + ML + RL |
|--------|-------------------:|---------------------:|
| Test matches | 240 | 240 |
| Value bets placed | 37 | 45 |
| Strike rate | 54.1% | 48.9% |
| **ROI** | **+17.9%** | **−2.3%** |
| Final bankroll ($1,000 start) | $1,179 | $977 |
| Sharpe ratio | 0.37 | 0.06 |
| Max drawdown | 16.1% | 25.8% |
| Profit factor | 1.18 | 0.98 |
| Avg edge (selected bets) | 25.3% | 20.6% |
| Avg CLV | −0.12% | −0.18% |

*Full details: [`backtests/results/metrics.txt`](backtests/results/metrics.txt) · [`metrics_ml_rl.txt`](backtests/results/metrics_ml_rl.txt) · bets logs in the same folder.*

**An honest finding:** on this canonical run the *simpler* configuration (PoissonElo + fractional Kelly) is the better performer. Adding the ML and RL layers did not improve returns — 45 bets is a very small sample, and a single backtest path is dominated by variance (the [Monte-Carlo simulation](#-the-1m-question--monte-carlo-simulation) is the honest summary). The layered pipeline exists to demonstrate the full ML stack, not because more layers always win.

### Model quality on the held-out test split (all 240 matches, not just bets)

| Metric | Model | Baseline |
|--------|------:|---------:|
| Accuracy | **54.6%** | 46.7% (most common outcome) |
| Log-loss | **0.98** | 1.10 (random 3-class) |
| Brier score | **0.577** | 0.667 (uniform) |

The model genuinely learns team strength: better than random on 240 matches it never saw.

### Visualisations (generated by the code, from the actual bets log)

![Backtest analysis](assets/backtest_analysis_ml_rl.png)

### 💰 The $1M question — Monte-Carlo simulation

`demo/simulation.py` trains the full pipeline, then runs **25 independent forward paths of 1,200 matches** (~3.3 years) with the trained model, starting each at **$1,000,000**. The default staking policy is **flat $10K per bet** — the variance-minimising policy found by the 100-trial staking stress test (`scripts/08_staking_stress_test.py`); the old quarter-Kelly default can be restored with `--policy kelly`.

| Statistic | Value |
|-----------|------:|
| Expected final bankroll (mean) | $1.44M (+43.6%) |
| Median final bankroll | $1.38M |
| P(finish in profit) | 88% |
| 90% range | $0.84M – $1.94M |
| Median CAGR | +10.2% / yr |

**Read this carefully:** flat staking removes the compounding boom/bust — no trial ever fell below $802K, and the median is positive. The mean is only slightly above the median (no fat-tail illusion). This is the honest picture of a modest-edge strategy: **positive and relatively stable**, but a 12% chance of finishing in the red even inside a calibrated synthetic world. It is **not** a claim about real markets.

![Simulation](demo/output/simulation_1m.png)

### The same $1M, but with a DYNAMIC thinking layer — confidence-aware

[`demo/simulation_dynamic.py`](demo/simulation_dynamic.py) re-runs the $1M question with `models/dynamic_thinking.py` — every decision is made adaptively: the model fuses the **public-vs-sharp market split** (opening vs closing line divergence, a classic hidden signal), re-weights model-vs-market from its own rolling Brier, shrinks stakes by model/market disagreement and drawdown, and switches to a low-risk survival mode below 10% of the start. The layer is **confidence-aware**: its confidence (how far the top outcome sits above the uniform ⅓) weights the calibration blend, gates immediate base-model refits when rolling confidence decays, and scales stakes up only when it is genuinely more sure. 12 trials × 1,200 matches, same streams, four policies:

| Policy | Mean | Median | P(profit) | 90% range | Median CAGR |
|--------|-----:|-------:|:---------:|:---------:|------------:|
| flat ($10K/bet) | $1.37M | $1.36M | 75% | $0.80M–$1.96M | +9.8% |
| quarter-Kelly | $6.40M | $0.27M | 25% | $0.01M–$33.0M | −32.7% |
| **dynamic (confidence-aware)** | $1.29M | $1.21M | 92% | **$1.01M–$1.61M** | +6.1% |
| dynamic v1 (original layer) | $1.28M | $1.25M | 100% | $1.14M–$1.46M | +7.0% |

**Honest reading — is the upgrade better?** The confidence-aware layer keeps the dynamic signature — tight 90% range ($1.01M–$1.61M, no path below $957K), 92% P(profit), model-vs-market weight converging to ~0.48 — and now **refits its base model ~13.5× per trial (1.5 of them confidence-gated)** when its rolling confidence decays. Head-to-head against the *previous* (v1) layer on identical streams, the two are statistically indistinguishable at 12 trials: v1 had a marginally tighter tail and 100% P(profit) in this sample, the confidence-aware layer a slightly higher mean. So the honest answer is: the confidence-aware upgrade is **not a clear return winner in a static synthetic world** — its value is behavioural (it re-learns when it loses confidence and scales risk with certainty), and the real-data experiments ([CLV](docs/09_real_walkforward_simulation.md), [hidden signals](#hidden-signals-on-real-data--consensus-dispersion-and-the-sharp-vs-public-split)) are where market value actually shows. Full table: [`docs/12_dynamic_thinking.md`](docs/12_dynamic_thinking.md). Watch the decision trace live in [`demo/output/simulation_live_dynamic.mp4`](demo/output/video_player.html).

---

## Deep learning — PyTorch NN + TensorFlow hybrid

Two deep models are trained on **all** synthetic training data and then tested on **real** match data they have never seen:

- **PyTorch MLP** — 4 engineered features → 64 → 64 → 3 softmax (CPU, seeded, reproducible).
- **TensorFlow hybrid** — an MLP that fuses the 4 features with the PoissonElo model's probability outputs (a genuine statistical + deep hybrid).

Features for the real leagues come from the *previous* real season (Elo + shifted form), so the test measures the learned **feature → probability mapping**, not cold-starting on unknown teams.

```bash
python scripts/04_deep_learning_transfer.py --offline
```

**Accuracy on real 2025/26 matches** (models trained on synthetic data only):

| Method | La Liga | Premier League (unseen) |
|--------|--------:|------------------------:|
| Market (real bookmaker odds) | **54.5%** | **48.9%** |
| sklearn Random Forest | 47.9% | 42.6% |
| sklearn Logistic Regression | 48.2% | 40.3% |
| sklearn Ridge classifier | 48.2% | 41.6% |
| sklearn Gradient Boosting | 47.6% | 41.6% |
| TF hybrid (NN + PoissonElo) | 42.4% | 37.4% |
| PyTorch NN | 44.0% | 41.6% |
| Majority / base rate | 48.9% | 42.6% |

**Honest reading:** synthetic-trained models do **not** beat the real market — the bookmaker remains the strongest predictor. Simple tree/linear models generalise out-of-distribution better than deep nets, and the cold-start row (no team info) collapses to the base rate. Conclusion: the synthetic world validates the *methodology*; real deployment requires retraining on real data (full numbers in [`docs/04_deep_learning_transfer.md`](docs/04_deep_learning_transfer.md)).

**What happens when the same nets are trained on REAL data?** [`scripts/06_deep_learning_real.py`](scripts/06_deep_learning_real.py) trains both deep models on real La Liga history (and on synthetic, for contrast) and iterates on weaknesses. Real training data lifts the nets from ~50.5% to **52.1% on unseen La Liga** and from 43.4% to **47.6% on unseen Premier League** (vs 48.9%/42.6% majority baseline). The iteration loop is fully instrumented: temperature calibration cut ECE 0.11→0.04, early stopping + dropout cut the train→test overfitting gap from +0.13/+0.20 to +0.02/+0.06, and a class-weighting experiment proved the model's failure to predict draws is a feature-information limit, not a training bug (forcing draws destroys accuracy). Full root-cause analysis in [`docs/07_deep_learning_real.md`](docs/07_deep_learning_real.md).

---

## Real-data validation — season-by-season backtest

Random splits are forbidden for temporal data. [`scripts/05_season_backtest.py`](scripts/05_season_backtest.py) runs an **expanding-window** backtest on five real La Liga seasons (2021/22 → 2025/26) — every test season is genuinely unseen, and features for match *i* use only matches strictly before *i* (running Elo + last-5 form, computed online).

```bash
python scripts/05_season_backtest.py --offline
```

**Accuracy by test season** (models trained only on earlier seasons):

| Method | 22/23 | 23/24 | 24/25 | 25/26 |
|--------|------:|------:|------:|------:|
| Majority / base rate | 47.9% | 44.0% | 44.5% | 48.9% |
| PoissonElo model | **54.7%** | 54.5% | 54.2% | **53.4%** |
| Ridge classifier | 53.2% | **54.5%** | **55.0%** | 51.6% |
| Gradient Boosting | 50.3% | 49.2% | 51.6% | 51.8% |
| Random Forest | 52.6% | 53.9% | 53.2% | 51.8% |

*(Full per-season metrics incl. log-loss, Brier and calibration ECE: [`backtests/results/season_backtest_results.csv`](backtests/results/season_backtest_results.csv).)*

**Cross-league transfer** (train all 21/22–24/25, test the other league's 25/26): La Liga-trained models score ~47–48% on unseen Premier League matches (vs 42.6% baseline); Premier League-trained models score ~49–50% on unseen La Liga (vs 48.9% baseline). Transfer is positive but smaller than within-league gains.

**Why 55%, not 65%?** On real three-way football data the accuracy ceiling is ~50–58% — draws are nearly unpredictable, and the published literature lands in the same range. Our models beat the majority baseline by 5–10 points with stable log-loss and calibration (ECE 0.03–0.09), which is a realistic, defensible result (see [`docs/05_season_backtest.md`](docs/05_season_backtest.md) and [`docs/06_research_notes.md`](docs/06_research_notes.md)).

**Live replay of real seasons.** [`demo/real_simulation.py`](demo/real_simulation.py) replays **all five La Liga seasons (2021/22–2025/26) with $1M and only point-in-time knowledge** — the fixture list and pre-match odds are known, results are revealed chronologically, Elo/form update online, quarter-Kelly staking with a 15% daily cap and a −90% drawdown switch to low-risk spread betting (`--multi` runs both books):

| Book | Mean ROI (5 seasons) | Median ROI | Positive seasons | Avg CLV vs sharp line |
|------|---------------------:|-----------:|:----------------:|----------------------:|
| **B365** (soft) | +3.9% (95% CI −2.4..+10.2) | +1.1% | 3/5 | **−2.54% (t=−17.6, p<0.001)** |
| **Pinnacle** (sharp) | +9.6% (95% CI −1.4..+20.6) | +11.1% | 3/5 | **+2.84% (t=16.4, p<0.001)** |

**The decisive finding:** betting at soft B365 prices the model systematically pays ~2.5% per bet more than the sharp line (negative CLV, t=−17.6) and barely breaks even; betting at sharp Pinnacle prices it *beats* the B365 line by +2.84% per bet (t=16.4) with a +9.6% mean ROI. The 95% CIs on ROI still include zero — five seasons is a small sample — but the CLV difference is overwhelming and quantifies exactly how much the soft bookmaker margin costs (details in [`docs/09_real_walkforward_simulation.md`](docs/09_real_walkforward_simulation.md)).

---

## Adaptive model — one model, many leagues and sports

A model trained on one league (or sport) reflects that league's statistics: home-advantage size, scoring rate, draw frequency. Point it at a *different* league and a frozen model degrades. [`models/adaptive_model.py`](models/adaptive_model.py) keeps learning online: Elo and rolling form update after every match (only past matches, never the future), and the Gradient-Boosting layer **refits on a rolling window** when scheduled or when its rolling Brier drifts — the same pipeline works on Serie A, La Liga, the Premier League, or a different sport because its features are league-agnostic (Elo difference, rolling goals, form points; no team identity).

```bash
python scripts/10_adaptive_transfer.py --offline   # downloads Serie A on first run
```

Trained on **real Serie A 2020/21–2023/24** (1,520 matches), then walked point-in-time over unseen matches:

| Target (unseen) | Adaptive | Static (frozen ML) | Majority |
|-----------------|---------:|-------------------:|---------:|
| **Serie A 2025/26** (within-league) | **50.0%** (1st half 48.4% → 2nd 51.6%) | 48.2% | 38.9% |
| La Liga 2025/26 (cross-league) | 45.0% | 48.2% | 48.9% |
| Premier League 2025/26 (cross-league) | 42.6% | 44.2% | 42.6% |
| Basketball-like league (cross-sport) | 78.2% · Brier **0.422** | 80.8% · Brier 0.538 | 54.9% |

**Honest reading:** (1) **Within-league, adaptation pays** — +1.8 pts over a frozen model, and it improves *through* the season as it adapts (48.4% → 51.6%). (2) **Cross-league cold transfer is hard** — refitting on a short window of a foreign league adds noise before enough data accumulates, so a frozen model is safer for the *first* season; this is a documented limitation, not a bug. (3) **Cross-sport, both models transfer via Elo**, but the adaptive model is dramatically better calibrated (Brier 0.422 vs 0.538, log-loss 0.746 vs 0.915) — exactly what matters for betting. Full numbers: [`docs/10_adaptive_transfer.md`](docs/10_adaptive_transfer.md).

## State-space models (LSTM / GRU) — and is the database worth more than the model?

A match is not an isolated event: a team arrives with a *state* built from its recent matches. [`models/lstm_model.py`](models/lstm_model.py) tests architectures that model that changing state directly — an **LSTM/GRU over each team's rolling last-8 match sequence** (the hidden state *is* the team's learned evolving form), fused with Elo difference, bookmaker odds and form — under the identical point-in-time protocol:

```bash
python scripts/11_lstm_state_test.py --offline
```

**Head-to-head on unseen 2025/26 matches (trained on real Serie A 2020/21–2023/24):**

| Method | Serie A 25/26 | La Liga 25/26 | Premier League 25/26 |
|--------|--------------:|--------------:|---------------------:|
| PoissonElo | **51.8%** | **50.8%** | 46.6% |
| Gradient Boosting | 50.3% | 49.7% | 46.8% |
| Adaptive (online refits) | 49.2% | 46.6% | 44.5% |
| **LSTM (rich features)** | 50.5% | 47.9% | 46.3% |
| **GRU (rich features)** | 49.2% | 48.7% | **47.9%** |
| LSTM thin (goals only) | 50.5% | 48.2% | 48.4% |
| Majority / base rate | 39.0% | 48.9% | 42.6% |

**Honest reading:** on ~1,500 training matches the sequence architecture **matches but does not beat** the simpler feed-forward baselines — PoissonElo stays best within-league, and GRU is competitive cross-league. That is a legitimate, literature-consistent result: sequence models need far more data than one league's four seasons to justify their parameters, and the *testing protocol* (point-in-time, expanding window) is what makes the comparison trustworthy.

**Your thesis, tested directly — does the database matter more than the model?** The *same* LSTM is trained on 1 vs 2 vs 3 real leagues (Serie A → +La Liga → +EPL) and tested on the same Serie A 25/26: 50.0% → 49.2% → 50.5%. **More same-sport leagues barely moves accuracy** — because the added rows carry the same information as the ones you already have. What moves the needle is *information variety*: real vs synthetic data (worth +1.6–4.2 pts, `scripts/06`), pre-match odds and rich match features, and a rigorous testing protocol. The database matters — but as *feature/content richness*, not raw row count. Full numbers: [`docs/11_lstm_state_test.md`](docs/11_lstm_state_test.md).

## Hidden signals on real data — consensus, dispersion, and the sharp-vs-public split

Real football-data CSVs carry up to a dozen bookmakers per match. [`scripts/12_hidden_signals.py`](scripts/12_hidden_signals.py) treats the market itself as a signal source on real **Serie A 2025/26** (trained 2020/21–2023/24, point-in-time):

- **Consensus & dispersion.** The average of all books' implied probabilities (the consensus) is itself a predictor; their disagreement (dispersion) tells you how hard a match is to price. After a home/away mapping bug was caught and fixed (the shared `CLASS_MAP` indexes `[away, draw, home]` but the consensus vector is `[home, draw, away]` — silently swapping every comparison below the chance floor), the corrected numbers are sane: **52.1% of low-dispersion and 56.3% of high-dispersion matches** are correctly classified by consensus — above the 33% chance floor, and, interestingly, high-dispersion matches were *easier*, not harder, to call.
- **Sharp-vs-public split — the money finding.** When the sharp Pinnacle line diverges from the public B365 line, betting the outcome the sharp line is most bullish on *at the public price* yields **CLV +1.16% per bet (n=200, t=4.72, p<0.001)** — positive closing-line value on real matches, the same mechanism the La Liga replay showed at scale.
- **The dynamic thinking layer on real 2025/26** fused with real multi-book signals finished −5.5% ROI (18 bets) — real prices are genuinely hard to beat; the CLV experiment above is where the value signal actually lives.

Full numbers: [`docs/13_hidden_signals.md`](docs/13_hidden_signals.md) and [`backtests/results/hidden_signals_results.csv`](backtests/results/hidden_signals_results.csv).

---

## Prediction interface

```bash
python predict_match.py --home "Real Madrid" --away "Barcelona"
python predict_match.py --home "Arsenal" --away "Liverpool" --league E0
```

Trains the PoissonElo + Gradient-Boosting layers on cached real history and prints a structured prediction — probabilities, chosen outcome, confidence/risk, Elo difference, recent form, league baselines, and data freshness:

```text
  Real Madrid  vs  Barcelona   [La Liga]
  Home win :  43.3%
  Draw     :  26.8%
  Away win :  29.9%
  ML prediction : Real Madrid (home win)
  Confidence    : 43% (margin over next outcome 13.3%) -> Low risk
  Elo diff      : -0 (home 1747, away 1747)
  ...
```

---

## Installation

Tested on **Windows (Python 3.14)** and clean-venv installs; macOS/Linux use the same commands with `source .venv/bin/activate`.

```bash
git clone https://github.com/ojasvigoel598/qunatative-ML.git
cd qunatative-ML

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Core requirements: `pandas, numpy, scipy, statsmodels, scikit-learn, seaborn, matplotlib, requests, openpyxl`. The deep-learning experiments additionally need PyTorch and TensorFlow — install with `pip install -r requirements-deep.txt` (optional; the core pipeline does not need them).

---

## Usage

```bash
# 1) Full pipeline: PoissonElo + ML + RL  (trains, backtests, saves metrics + plots)
python run_full_ml_rl.py

# 2) Base pipeline: PoissonElo + fractional Kelly
python run_full_project.py

# 3) End-to-end narrated demo (prints every stage, saves to demo/output/)
python demo/demo_end_to_end.py

# 4) "$1M invested — what happens?" Monte-Carlo simulation
python demo/simulation.py --trials 25 --matches 1200

# 5) Deep-learning transfer: PyTorch NN + TF hybrid, tested on real La Liga + EPL
python scripts/04_deep_learning_transfer.py --offline     # needs requirements-deep.txt

# 6) Real-data, time-aware validation: season-by-season + cross-league
python scripts/05_season_backtest.py --offline

# 7) Predict a fixture (real history, cached)
python predict_match.py --home "Real Madrid" --away "Barcelona"

# 8) Real walk-forward simulation: replay La Liga 2022/23 with $1M, point-in-time knowledge
python demo/real_simulation.py                # + real CLV vs Pinnacle closing line

# 9) Full metrics report: every tracked build + statistical uncertainty (CIs, t-tests)
python scripts/07_full_metrics_report.py

# 10) 100-trial staking stress test: ruin probability vs stake caps and Kelly
python scripts/08_staking_stress_test.py --trials 100

# 11) Interactive metrics dashboard (self-contained HTML)
python scripts/09_make_dashboard.py          # -> docs/dashboard.html

# 12) Verification suite (12 tests)
python -m pytest tests/ -v

# 13) Explain the ML pipeline end-to-end (executed notebook, open in Jupyter/GitHub)
notebooks/01_explained_ml_pipeline.ipynb

# 14) Adaptive cross-league / cross-sport transfer (Serie A -> La Liga -> EPL)
python scripts/10_adaptive_transfer.py --offline

# 15) State-space models (LSTM/GRU) + database-vs-model test
python scripts/11_lstm_state_test.py --offline

# 16) Hidden signals on real data (consensus, dispersion, sharp-vs-public CLV)
python scripts/12_hidden_signals.py --offline

# 17) $1M simulation driven by the dynamic thinking layer
python demo/simulation_dynamic.py

# 18) Render the demo videos (needs: pip install imageio imageio-ffmpeg)
python demo/make_simulation_video.py      # -> demo/output/simulation_live_flat.mp4
python demo/make_dynamic_video.py         # -> demo/output/simulation_live_dynamic.mp4
python demo/make_serie_a_video.py --offline   # -> demo/output/serie_a_live.mp4
#    watch all three in demo/output/video_player.html

# 19) One-command pipeline: regenerate EVERY artifact + verify
bash ci_report.sh            # full (deep-learning experiments included)
bash ci_report.sh --fast     # skip the long deep-learning experiments
```

Outputs land in `backtests/results/`:

```
backtest_bets_log_ml_rl.csv      # every bet: odds, stake, edge, outcome, CLV
metrics_ml_rl.txt                # summary metrics
backtest_analysis_ml_rl.png      # equity curve, edge/CLV distributions, P/L bars
backtest_summary_ml_rl.png       # one-page results card
```

**Reproducibility:** every script defaults to `seed=42`. Run any command twice — identical data, models, bets, and metrics.

---

## Project structure

```
qunatative-ML/                    # repository root (this project)
├── pipeline.py                    # shared engine: data, training, backtest, metrics
├── run_full_project.py            # CLI: base pipeline (PoissonElo + Kelly)
├── run_full_ml_rl.py              # CLI: full pipeline (PoissonElo + ML + RL)
├── predict_match.py               # CLI: predict Team A vs Team B
├── models/
│   ├── poisson_elo_model.py       # Layer 1: Elo + Poisson regression
│   ├── ml_layer.py                # Layer 2: calibrated Gradient Boosting
│   ├── rl_staking_agent.py        # Layer 3: Q-learning staking agent
│   ├── nn_model.py                # PyTorch MLP (deep-learning layer)
│   ├── tf_hybrid.py               # TensorFlow hybrid (MLP + PoissonElo)
│   ├── adaptive_model.py          # dynamic model: online Elo/form + drift-triggered refits
│   ├── lstm_model.py              # state-space model: LSTM/GRU over rolling team sequences
│   └── dynamic_thinking.py        # decision layer: market-split signal + adaptive blend + risk-aware staking
├── demo/
│   ├── demo_end_to_end.py         # narrated full-project demo
│   ├── simulation.py              # $1M Monte-Carlo simulation (synthetic world)
│   ├── simulation_dynamic.py      # $1M simulation driven by the thinking layer
│   ├── real_simulation.py         # real walk-forward: La Liga, point-in-time + CLV
│   ├── make_simulation_video.py   # renders the $1M simulation as an mp4
│   ├── make_dynamic_video.py      # renders the dynamic-thinking trial as an mp4
│   ├── make_serie_a_video.py      # renders a real Serie A 25/26 replay as an mp4
│   └── output/
│       ├── simulation_live_*.mp4 · serie_a_live.mp4 · video_player.html
├── notebooks/
│   └── 01_explained_ml_pipeline.ipynb   # ML pipeline explained end-to-end
├── scripts/
│   ├── 01_data_ingestion.py       # real-data download (football-data.co.uk)
│   ├── 02_backtest.py             # backtest CLI
│   ├── 03_generate_assets.py      # regenerates README images
│   ├── 04_deep_learning_transfer.py  # PyTorch + TF hybrid transfer experiment
│   ├── 05_season_backtest.py      # real-data season-by-season backtest
│   ├── 06_deep_learning_real.py   # deep nets on real vs synthetic data + iteration
│   ├── 07_full_metrics_report.py  # consolidated metrics + uncertainty (CIs, t-tests)
│   ├── 08_staking_stress_test.py  # 100-trial staking policy sweep (ruin analysis)
│   ├── 09_make_dashboard.py       # interactive docs/dashboard.html generator
│   ├── 10_adaptive_transfer.py    # adaptive model: cross-league + cross-sport transfer
│   ├── 11_lstm_state_test.py      # LSTM/GRU state-space test + database-vs-model
│   └── 12_hidden_signals.py       # consensus / dispersion / sharp-vs-public CLV on real data
├── data/real_data.py              # shared multi-league loader (SP1 / E0 / I1, rich columns)
├── tests/test_pipeline.py         # 12 verification tests
├── data/processed/                # historical_matches.csv (auto-generated)
├── data/real/                     # cached real seasons (La Liga, EPL, Serie A)
├── data/real_data.py              # shared multi-league loader (SP1 / E0 / I1)
├── backtests/results/             # metrics, bets logs, transfer + season results
├── docs/                          # data sources, architecture, experiments, research
│   └── dashboard.html             # interactive metrics dashboard (self-contained)
├── assets/                        # README images
├── requirements-deep.txt          # optional PyTorch / TensorFlow extras
├── ci_report.sh                   # one-command regenerate-everything + verify pipeline
└── predictions/ · logs/           # forward P&L tracking workflow + template
```

---

## Methodology

**World model.** Each team has a latent strength `s ~ N(0,1)`. Expected goals follow a Poisson process with home advantage (`λ_home = 1.6·e^{0.22·Δs}·1.12`, `λ_away = 1.3·e^{−0.22·Δs}`); results follow. Bookmaker odds are the *true* probabilities → fair odds → multiplied by a margin (`~U(5%, 8%)`) and the favourite–longshot bias (`p_bookie ∝ p_true^0.88`, a real documented market effect: bookies overprice longshots). Closing odds are drawn independently with less noise, which makes **CLV** — how the line moved after you bet — meaningful and centred near zero.

**Model.** Elo ratings are updated sequentially (match *i* sees only matches `0..i−1`). Two Poisson regressions (`statsmodels`) map Elo → expected goals; the 0–8 goal grid is summed into outcome probabilities. A Gradient-Boosting classifier on the same Elo plus *shifted* rolling form adds a flexible second opinion; its probabilities are sigmoid-calibrated with internal cross-validation. The final prediction is the average of the two.

**Betting.** `edge = p × odds − 1` per outcome; a bet fires when the best edge exceeds **3%** on an outcome the model is **≥ 40%** confident about — the probability floor is a deliberate guard against the *winner's curse* (longshot probabilities carry too much relative estimation error). The Q-learning agent (state = edge/bankroll bins, action = Kelly multiplier ∈ {0, 0.5, 1.0, 1.5}×) sizes stakes and is trained **only** on validation-split bets.

**Evaluation.** The test split is touched exactly once. Metrics are computed from the full bets log: ROI on starting bankroll, strike rate, per-bet Sharpe annualised by actual bet frequency (not an arbitrary 252), peak-to-trough drawdown, profit factor, and average CLV. Model quality is also scored on *all* test matches (accuracy, log-loss, Brier) so the headline isn't just "the bets won".

---

## Validation

- **12 automated tests** (`python -m pytest tests/ -v`) verify: no same-team fixtures, result/goal consistency, no target leakage in rolling features, Poisson predictions sum to 1 and beat random baselines, the ML layer distinguishes teams, RL stakes stay bounded, backtest bookkeeping is internally consistent, and full **reproducibility** (two identical runs → identical bets and metrics).
- **The CLV bug is fixed and tested**: the original `clv = (odds/odds − 1) × 100` always returned 0; CLV now uses real closing odds and the test asserts it is non-trivial.
- **No data leakage**: chronological split, sequential Elo, shifted form features — covered by a dedicated test.
- **Reproducible**: seed 42 everywhere; identical outputs on every machine.

---

## Demo videos

Two **real `.mp4` walkthroughs** are rendered from the actual run outputs (matplotlib + a bundled ffmpeg — no AI-generated or fabricated footage):

| Video | What it shows | File |
|-------|---------------|------|
| **$1M simulation** | The trained model betting through 1,200 synthetic matches: equity curve, every bet as a green/red marker sized by stake, an *ML THINKING* panel (probability bars, odds, edge, stake, WIN/LOSS flash), a BIGGEST-WIN highlight and a speedrun | `demo/output/simulation_live_flat.mp4` (50 s) |
| **Real Serie A 25/26 replay** | The adaptive model replaying the real 2025/26 season point-in-time with $1M: real teams, real B365 odds, results revealed chronologically | `demo/output/serie_a_live.mp4` (42 s) |

Watch both in [`demo/output/video_player.html`](demo/output/video_player.html). Render them yourself with:

```bash
pip install imageio imageio-ffmpeg
python demo/make_simulation_video.py
python demo/make_serie_a_video.py --offline
```

A frame-accurate recording plan with narration for a screen-recorded version (setup → architecture → execution → methodology → results → simulation) is in [`docs/03_demo_video_script.md`](docs/03_demo_video_script.md).

---

## Limitations

1. **Synthetic data by default.** Odds come from the same true distribution the model learns. The synthetic backtest proves the *pipeline mechanics*; it says nothing about real-world profitability. Real-data experiments (`scripts/04`, `scripts/05`, `predict_match.py`) show the models beat the majority baseline but **not** the real bookmaker — consistent with the ~50–58% accuracy ceiling in the published football-prediction literature.
2. **Deep nets do not transfer from synthetic to real.** The transfer experiment is honest: simple tree/linear models generalise better than the PyTorch/TensorFlow nets out-of-distribution. Retraining on real data is the path to deployment.
3. **The average edge looks large (~20–25%).** Selection inflates it: the model bets exactly where it disagrees with the bookmaker most, and the synthetic bias is deliberately strong. Real markets are far tighter; treat the edges as relative, not absolute.
4. **Variance.** A single 240-match backtest is one path of a noisy process. The simulation is the honest summary: under the variance-minimising flat staking the 90% range is $0.84M–$1.94M, and even then 12% of paths finish in the red — under the old quarter-Kelly default, 22% of 100-trial paths dipped below $100K.
5. **Simple features.** No injuries, xG, opponent-specific form, weather, referee, or live-market data. Real systems add these.
6. **RL agent is a demo-scale learner** — a tabular Q-table over ~40–100 validation experiences, and in the canonical run it did **not** improve returns over plain fractional Kelly. It demonstrates the concept; it is not a production bankroll manager.

---

## Future improvements

- **Retrain the deep nets on real data** — the transfer experiment shows the mapping is learnable (models beat the baseline cross-league); real training data should close the gap to the market.
- Add the Dixon–Coles low-score dependence correction (τ) and real market odds as features.
- Richer features (xG, shots, standings, injuries) and stronger learners (LightGBM, stacking).
- Forward P&L logging for a live track record (`predictions/forward_prediction_workflow.md`).
- Calibration curves + reliability diagrams in the notebook.
- A dashboard (Streamlit) for live equity, CLV, and bet history.

---

## Technologies

Python · pandas · NumPy · SciPy · statsmodels (Poisson regression) · scikit-learn (Gradient Boosting, Ridge, calibration) · **PyTorch** (neural network) · **TensorFlow** (hybrid model) · Matplotlib / Seaborn (visualisation) · requests (data ingestion) · Jupyter (notebook) · pytest (verification).

---

## License

[MIT](LICENSE) — free to use, learn from, and build on.

*This project is for educational and portfolio purposes. Nothing here is financial or gambling advice. Always gamble responsibly.*
