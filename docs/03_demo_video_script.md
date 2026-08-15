# Demo Video — Complete Recording Plan & Narration Script

> **Status: plan + script (no video file was rendered in this environment).**
> This document is a frame-accurate plan for recording the demo yourself
> (OBS Studio or any screen recorder). It walks the *real, running project* —
> no slides, no fabricated output.

**Target length:** ~6 minutes. **Format:** 16:9, 1080p. **Audio:** narration over screen capture.

---

## Suggested README embed

After you have rendered the video, upload it (e.g. YouTube) and embed it in the README hero section:

```markdown
[![Demo video](https://img.youtube.com/vi/REPLACE_WITH_ID/0.jpg)](https://www.youtube.com/watch?v=REPLACE_WITH_ID)
```

Until a video exists, do **not** add the embed — the README must never claim a video that was not made.

---

## Global setup for the recording

| Item | Value |
|------|-------|
| Working directory | repo root (`sports_betting_model/`) |
| Python | `python -m venv .venv` then activate (Windows: `.venv\Scripts\activate`) |
| Dependencies | `pip install -r requirements.txt` |
| Font size | Terminal 16pt+, light theme for readability |
| Terminal width | ≥ 120 columns |

---

## Recording sequence (9 sections)

### 1. Introduction (0:00–0:35)
**On screen:** repo README (GitHub or local render), scrolled slowly.

**Narration:**
> "This is a quantitative sports-betting model for football. It prices matches
> with a Poisson + Elo hybrid, adds a Gradient-Boosting machine-learning layer,
> sizes stakes with a Q-Learning reinforcement-learning agent, and backtests
> the whole strategy with honest metrics: ROI, Sharpe, max drawdown and closing
> line value. The goal is to demonstrate a complete, reproducible ML pipeline —
> from raw match data to a fully evaluated betting strategy."

### 2. Repository structure (0:35–1:05)
**Command:** `tree -L 2` (or `ls -R` on Windows).

**Narration:**
> "The repository is split into clear layers: `models/` holds the three model
> implementations, `pipeline.py` is the shared engine that ties data, training
> and backtesting together, `demo/` contains the end-to-end demo and the $1M
> Monte-Carlo simulation, `notebooks/` has an explained walkthrough of the ML
> pipeline, and `backtests/results/` receives every run's bets log, metrics and
> charts. Everything is seeded, so every run is reproducible."

### 3. Setup (1:05–1:35)
**Commands:**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

**Narration:**
> "Setup is a standard Python venv plus one requirements file. The only
> dependencies are pandas, numpy, scipy, scikit-learn, statsmodels, seaborn
> and matplotlib."

### 4. Code architecture (1:35–2:20)
**On screen:** `models/poisson_elo_model.py`, `models/ml_layer.py`, `models/rl_staking_agent.py`, `pipeline.py` opened side by side (or tabs).

**Narration:**
> "Three model layers, one engine. `PoissonEloModel` trains Elo ratings and two
> Poisson regressions, then converts the score grid into win/draw/away
> probabilities. `MLFootballPredictor` is a calibrated Gradient Boosting
> classifier on the same Elo features plus shifted rolling form — the shift
> guarantees no target leakage. `QLearningStakingAgent` learns stake sizes as
> Kelly multipliers from realized validation bets. `pipeline.py` orchestrates
> the train/validation/test split, the edge calculation — model probability
> times bookie odds minus one — and the backtest loop."

### 5. Main execution (2:20–3:20)
**Command:**
```bash
python run_full_ml_rl.py
```
**On screen:** the full run, captured top to bottom.

**Narration:**
> "This is the entire project running end to end. It generates the synthetic
> football world — teams with latent strength, Poisson goals, bookmaker odds
> built from true probabilities plus margin and the favourite-longshot bias —
> trains all three layers on the train split, discovers bets on the validation
> split to train the staking agent, and then runs the final backtest on the
> never-touched test split. Watch the metrics file and the four charts appear
> in `backtests/results/`."

### 6. Core methodology (3:20–4:05)
**On screen:** the notebook `notebooks/01_explained_ml_pipeline.ipynb` scrolled through the model sections.

**Narration:**
> "The methodology is worth spelling out. Elo gives each team a strength score;
> Poisson regressions turn that into expected goals; summing the score grid
> gives outcome probabilities. Edge is the difference between what the model
> thinks a bet is worth and what the bookmaker pays. We only bet when the edge
> is above 3% and the model is at least 40% confident — the probability floor
> avoids longshots, where small probability errors become huge relative errors,
> the so-called winner's curse. The ML layer's probabilities are sigmoid-
> calibrated because edge calculation is brutally sensitive to miscalibration."

### 7. Results (4:05–4:50)
**Commands:**
```bash
type backtests\results\metrics_ml_rl.txt     # Windows
cat backtests/results/metrics_ml_rl.txt      # macOS/Linux
```
**On screen:** metrics file, `backtest_analysis_ml_rl.png`, `backtest_summary_ml_rl.png`.

**Narration:**
> "These are the real numbers from the run you just watched. The full pipeline
> made 101 bets on the test split with a 47.5% strike rate, ended 32% up on a
> $1,000 starting bankroll, with a Sharpe ratio of 0.44 and a 28% maximum
> drawdown. The equity curve, edge distribution and CLV distribution are all
> generated from the actual bets log. Model accuracy on all 240 test matches
> is 53.8% versus a 46.7% baseline, with a log-loss of 0.98 versus 1.10 for a
> random classifier — the model genuinely learns team strength."

### 8. Validation (4:50–5:30)
**Commands:**
```bash
python -m pytest tests/ -v
python demo/simulation.py --trials 25 --matches 1200
```

**Narration:**
> "Results are only trustworthy if the methodology is sound. The test suite
> verifies there is no data leakage, that the CLV metric is computed from real
> closing odds, that predictions are calibrated, and that every run is
> reproducible. The simulation then answers the obvious question: if you
> invested one million dollars, what would happen? It runs twenty-five
> independent forward paths of 1,200 matches each with the trained model. The
> honest answer is a distribution, not a single number — inside this synthetic
> world the mean is strongly positive but the median path loses money (about
> 40% probability of finishing in profit), which is exactly the variance story
> a serious risk analysis should tell. That is a demonstration of the
> methodology, not a prediction of real-world returns."

### 8b. Deep learning + real-data validation (4:30–5:30, optional but recommended)
**On screen:** `python scripts/04_deep_learning_transfer.py --offline`, then
`python scripts/05_season_backtest.py --offline`, then the results tables;
`python predict_match.py --home "Real Madrid" --away "Barcelona"`.

**Narration:**
> "Beyond the synthetic world, the project includes deep-learning transfer
> experiments — a PyTorch neural network and a TensorFlow hybrid trained on
> all the data, then tested on real La Liga and Premier League matches the
> model has never seen. The honest result: on real football the market is
> still the strongest predictor, and simple models generalise better than
> deep nets out of distribution. And on real data, a season-by-season
> backtest beats the majority baseline by five to ten points — about the
> accuracy ceiling the literature reports for three-way football prediction."

### 9. Final result (5:30–6:30)
**On screen:** README hero + results section.

**Narration:**
> "To summarise: a complete, reproducible ML pipeline — Poisson + Elo, a
> calibrated gradient-boosting layer, a Q-learning staking agent, PyTorch and
> TensorFlow deep layers, an honest train/validation/test backtest with CLV,
> a season-by-season backtest on real league data, a Monte-Carlo simulation,
> a notebook that explains every step, and tests that verify the methodology.
> The repository is ready to be a portfolio project: runnable from a clean
> checkout, with real outputs generated by the code itself."

---

## Recording tips

1. **Record a fresh, clean run** — do not edit out failures; if something fails, fix it and record the fixed run. The video should show the project *actually working*.
2. Keep the terminal history clean (`clear` before each command).
3. Show `demo/demo_end_to_end.py` as a shorter alternative in section 5 if time is tight.
4. Silence system notifications; use a decent microphone.
5. After recording: trim dead air, add a caption of the bottom line ("$1M invested → median $X, P(profit) Y%") over the simulation chart.
