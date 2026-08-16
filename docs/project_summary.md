# Quantitative Sports Betting Model — 1-Page Summary

**Poisson + Elo hybrid · Gradient Boosting ML · Q-Learning staking · PyTorch NN + TensorFlow hybrid · Adaptive cross-league model — with honest, reproducible validation on synthetic and real data**

## Methodology
- Dynamic Elo ratings + Poisson regression for expected goals → outcome probabilities.
- Calibrated Gradient-Boosting ML layer on the same features (no leakage, team-aware).
- Value bets when **edge = p × odds − 1 > 3%**, restricted to confident outcomes (p ≥ 0.40).
- Quarter-Kelly staking, with a Q-Learning agent learning Kelly multipliers from validation bets.
- Full **CLV** tracking from independent closing odds.
- Chronological **train (65%) / validation (15%) / test (20%)** split — test is never tuned on.
- Deep learning: **PyTorch MLP** and **TensorFlow hybrid** (MLP fused with PoissonElo outputs) trained on all data and tested for cross-league transfer on real La Liga + Premier League.
- **Adaptive model**: league-agnostic features (Elo diff, rolling goals, form points) with online Elo/form updates and drift-triggered rolling refits — trained on real Serie A and tested on unseen Serie A 2025/26, La Liga, EPL and a cross-sport (basketball-like) world.
- Real-data time-aware validation: expanding-window season-by-season backtest (La Liga / Serie A / EPL) + cross-league transfer, with online (leakage-free) features.

## Backtest results (canonical run, fully reproducible)

| Metric | PoissonElo + Kelly | PoissonElo + ML + RL |
|--------|-------------------:|---------------------:|
| Test matches | 240 | 240 |
| Value bets | 26 | 23 |
| Strike rate | 42.3% | 47.8% |
| ROI | **−14.3%** | **−13.0%** |
| Final bankroll ($1,000 start) | $857 | $870 |
| Sharpe ratio | −0.48 | −0.39 |
| Max drawdown | 21.3% | 21.8% |
| Avg edge (selected bets) | 14.8% | 14.1% |
| Avg CLV | +0.01% | +0.18% |

**Model quality on the held-out test split (all 240 matches, not just bets):**
accuracy **54.6%** vs 46.7% baseline · log-loss **0.98** vs 1.10 (random) · Brier 0.577.

**Honest finding:** against a bookmaker with a real positive margin both configurations **lose** on this small sample (−14.3% / −13.0% ROI on 23–26 bets); the estimated edges (~14%) overstate realised returns (winner's curse). The model prices matches close to the bookmaker (test accuracy 54.6% vs 46.7% baseline) but not well enough to overcome the overround. The $1M Monte-Carlo simulation is the honest summary: median path **loses** ~6%, P(profit) 32%.

## Real-data experiments
- **Season-by-season (La Liga):** PoissonElo / Ridge ~53–55% accuracy on unseen seasons vs ~44–49% majority baseline; log-loss and calibration (ECE 0.03–0.09) stable.
- **Cross-league:** La Liga-trained models score ~47–48% on unseen Premier League (vs 42.6% baseline); EPL-trained models ~49–50% on unseen La Liga (vs 48.9%).
- **Deep-learning transfer:** synthetic-trained PyTorch/TF models do **not** beat the real bookmaker (market 54.5% La Liga / 48.9% EPL); simple models generalise better than deep nets out-of-distribution.
- Full tables: `backtests/results/transfer_results.csv`, `season_backtest_results.csv`.

## Adaptive transfer results (trained on real Serie A 2020/21–2023/24)
| Target (unseen) | Adaptive | Static (frozen ML) | Majority |
|-----------------|---------:|-------------------:|---------:|
| Serie A 2025/26 (within-league) | **50.0%** (48.4→51.6 across season) | 48.2% | 38.9% |
| La Liga 2025/26 | 45.0% | 48.2% | 48.9% |
| Premier League 2025/26 | 42.6% | 44.2% | 42.6% |
| Basketball-like league (cross-sport) | 78.2% · Brier 0.422 | 80.8% · Brier 0.538 | 54.9% |

Adaptation pays **within-league** (+1.8 pts, improving through the season); cold cross-league transfer favours the frozen model until enough foreign data accumulates; cross-sport both transfer via Elo but the adaptive model is far better calibrated (Brier 0.422 vs 0.538) — the property that matters for betting.

## Demo videos (rendered from real outputs)
- `demo/output/simulation_live_flat.mp4` — $1M simulation, live equity curve + ML THINKING panel + speedrun (50 s).
- `demo/output/serie_a_live.mp4` — real Serie A 2025/26 point-in-time replay, adaptive model, $1M start (42 s).
- Watch both in `demo/output/video_player.html`; render with `python demo/make_simulation_video.py` and `python demo/make_serie_a_video.py --offline`.

## Reproduce
```bash
python run_full_project.py      # base pipeline        -> backtests/results/metrics.txt
python run_full_ml_rl.py        # full ML + RL         -> backtests/results/metrics_ml_rl.txt
python demo/simulation.py       # $1M Monte Carlo      -> demo/output/simulation_1m.png
python scripts/04_deep_learning_transfer.py --offline  # PyTorch + TF hybrid transfer
python scripts/05_season_backtest.py --offline         # real-data season backtest
python scripts/10_adaptive_transfer.py --offline       # adaptive cross-league + cross-sport
python demo/make_simulation_video.py                   # $1M simulation video
python demo/make_serie_a_video.py --offline            # real Serie A 25/26 replay video
python predict_match.py --home "Real Madrid" --away "Barcelona"
python -m pytest tests/ -v      # verification suite (12 tests)
```

## Honest framing
- The default data is a **synthetic, calibrated world**; the backtest validates the *methodology*, not real-world profitability.
- On **real** data the models beat the majority baseline but not the bookmaker — consistent with the ~50–58% accuracy ceiling in the football-prediction literature (see `docs/06_research_notes.md`).
- The average edge of selected bets is **inflated by selection**; real markets are far tighter.
- A single backtest path is high-variance — always read the Monte-Carlo distribution.

## Tools
Python (pandas, numpy, scipy, statsmodels, scikit-learn, seaborn, matplotlib) · PyTorch · TensorFlow · football-data.co.uk (real data) · Jupyter · pytest

**Status:** complete, tested, reproducible — GitHub ready.
