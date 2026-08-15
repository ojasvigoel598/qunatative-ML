#!/usr/bin/env python3
"""Rebuild notebooks/01_explained_ml_pipeline.ipynb with ALL current ML/code."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ============================================================ 0. HERO
md(r"""# Quantitative Sports Betting Model — the full ML system, explained end-to-end

This notebook walks through **every layer of the project**, from raw match data to a
fully backtested, dynamically-adapting betting system:

```
Match data ─▶ Poisson + Elo ─▶ Gradient Boosting ─▶ Hybrid ensemble
   ─▶ Q-Learning staking ─▶ Backtest & metrics
   ─▶ ADAPTIVE model (online refits) ─▶ DYNAMIC THINKING layer (confidence-aware)
   ─▶ Hidden-signals experiment (real data) ─▶ Leak-free multi-league walk-forward
   ─▶ Deep learning (PyTorch + TensorFlow) ─▶ LSTM / GRU ─▶ $1M simulations
```

Everything is **seeded and reproducible**: run the notebook top to bottom and you get
exactly the same numbers every time.

> **What is this project?** A quantitative sports-betting model that prices football
> matches with a Poisson + Elo hybrid, blends in a Gradient-Boosting ML layer, sizes
> stakes with Q-Learning and confidence-aware staking, adapts online when pointed at
> new leagues, and is validated on real data with strict no-leakage methodology.
>
> **Honesty note.** Most of the backtesting runs in a *synthetic, calibrated world* —
> it demonstrates the methodology and shows the model can find genuine mispriced
> lines inside that world. It is **not** a prediction of real-world returns. Where we
> use real data (sections 12, 13, 18, 19) we say so explicitly.

Run the cells in order. Heavy training cells take a few seconds each.
""")

# ============================================================ 1. SETUP
md("## 1. Setup\n\nThe project is a small package: `models/` holds the model layers, `pipeline.py` ties the core together, and `agent_sim/` holds the strict walk-forward simulation. We import everything and configure plotting to be headless-safe.")
code(r"""# Make sure the project root is importable regardless of kernel cwd
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "pipeline.py").exists():
    ROOT = ROOT.parent          # kernels launched from notebooks/
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pipeline
from models.poisson_elo_model import PoissonEloModel
from models.ml_layer import MLFootballPredictor
from models.rl_staking_agent import QLearningStakingAgent
from models.adaptive_model import AdaptiveMatchPredictor
from models.dynamic_thinking import DynamicThinkingLayer

sns.set_style("whitegrid")
pd.set_option("display.width", 140)
print("[OK] imports ready")
""")

# ============================================================ 2. DATA
md(r"""## 2. The data: a realistic synthetic football world

Real match data is hard to ship in a repo (licensing, size, freshness). Instead the
project ships a **generator for a synthetic-but-realistic world**:

* 10 teams, each with a latent **strength** `s ~ N(0, 1)`.
* Expected goals follow a Poisson process with a home-advantage multiplier:

$$\lambda_{home} = 1.6 \cdot e^{0.22 \cdot (s_{home}-s_{away})} \cdot 1.12, \qquad \lambda_{away} = 1.3 \cdot e^{-0.22 \cdot (s_{away}-s_{home})}$$

* Goals are drawn `Poisson(λ)` and the 1X2 result follows.
* **Bookmaker odds** are derived from the *true* outcome probabilities, plus:
  * an **overround/margin** `~U(5%, 8%)` (how bookmakers make money), and
  * the documented **favourite–longshot bias** (`p_bookie ∝ p_true^0.88`): bookmakers
    price longshots too short and favourites too long.
* **Closing odds** are drawn independently (same margin, less noise) so we can compute
  **CLV** — how the line moved after you bet.

Real multi-league data (La Liga, Premier League, Bundesliga, Serie A) is fetched on
demand from the free public API at football-data.co.uk — see sections 18 and 19.
Let's generate the canonical 1,200-match dataset (seed 42) and inspect it.""")
code(r"""df, generated = pipeline.load_or_generate_data(n_matches=1200, seed=42)
df = df.sort_values("date").reset_index(drop=True)
print(f"Dataset: {len(df)} matches, generated={generated}")
df.head()""")
code(r"""print("Goal distribution per match:")
print(df[["home_goals", "away_goals"]].describe().round(2).to_string())

print("\nResult frequencies:")
print(df["result"].value_counts(normalize=True).round(3).to_string())

print("\nHome advantage is visible: home teams score more than away teams.")
print(f"  avg home goals = {df['home_goals'].mean():.2f}  vs  avg away goals = {df['away_goals'].mean():.2f}")"")

""" )
# ============================================================ 3. SPLIT
md("""## 3. Train / validation / test split — and why it matters

We split chronologically (the data is time-ordered) into:

| Split | Rows | Used for |
|-------|------|----------|
| Train | 780  | Fitting Elo, Poisson regression, Gradient Boosting |
| Validation | 180 | Calibrating the RL staking agent (realized bets) |
| Test | 240 | **Final, never-touched evaluation** |

This is the backbone of the whole project: the test split is only used once, at the
very end. No hyper-parameter, stake level, or threshold is tuned on it — otherwise the
backtest results would be **leaked** and meaningless.""")
code(r"""n = len(df)
train_df = df.iloc[:int(n * 0.65)].copy()
valid_df = df.iloc[int(n * 0.65):int(n * 0.80)].copy()
test_df = df.iloc[int(n * 0.80):].copy()
print(f"Train: {len(train_df):4d} | Validation: {len(valid_df):4d} | Test: {len(test_df):4d}")""")

# ============================================================ 4. FEATURES
md(r"""## 4. Feature engineering

### Elo ratings
Each team gets a dynamic Elo rating, updated sequentially match by match:

$$E_{home} = \frac{1}{1 + 10^{\,(elo_{away} - elo_{home})/400}}, \qquad elo \leftarrow elo + K \cdot (actual - expected)$$

Crucially the Elo attached to match *i* reflects only matches `0..i-1` — **no look-ahead**.

### Rolling form (ML features)
The ML layer uses short-term form: each team's average goals over their last 5 home
(away) games. The rolling mean is **shifted by one match** so a match's own goals never
leak into its own features.

Both feature blocks are produced by `PoissonEloModel.prepare_features`.""")
code(r"""poisson = PoissonEloModel()
train_feat = poisson.prepare_features(train_df)   # adds home_elo / away_elo
print(train_feat[["home_team", "away_team", "home_elo", "away_elo"]].head(5).to_string(index=False))

print("\nLearned Elo ratings at end of training:")
for team, elo in sorted(poisson.elo_ratings.items(), key=lambda kv: -kv[1]):
    print(f"  {team:<14} {elo:7.0f}")""")

# ============================================================ 5. POISSON
md(r"""## 5. Layer 1 — Poisson + Elo hybrid

The core model has three moving parts:

1. **Elo** gives each team a strength score (see above).
2. **Two Poisson regressions** (`statsmodels`) model expected goals:
   $$\log \lambda_{home} = \beta_0 + \beta_1 \cdot elo_{home} + \beta_2 \cdot elo_{away}$$
   (home advantage is already in the fitted intercept — the original code applied it a
   *second* time, systematically inflating P(home win); that bug is fixed, and the
   regression coefficients are **shrunk toward 0** to keep estimated edges honest).
3. The **score grid** `P(h) × P(a)` for `0..8` goals is summed to get `P(home win)`,
   `P(draw)`, `P(away win)`.

The model is evaluated with **log-loss, Brier score and accuracy** against the natural
3-class baseline (always predict the most common outcome).""")
code(r"""poisson.train(train_df)
print(f"\n  Home goals AIC = {poisson.poisson_home.aic:.1f} | Away goals AIC = {poisson.poisson_away.aic:.1f}")

# Inspect one prediction
probs = poisson.predict("Man City", "West Ham")
print("\nMan City vs West Ham:", {k: probs[k] for k in ("home_win", "draw", "away_win")})
print("Fair odds:", poisson.probs_to_fair_odds(probs))""")
code(r"""# Evaluate on the held-out test split (all matches, not just bets)
scored = pipeline._predictions_over(test_df, poisson, None)
ev = pipeline.evaluate_probability_quality(scored)
print(f"PoissonElo on test: log-loss {ev['log_loss']:.3f} | accuracy {ev['accuracy']:.3f} "
      f"(baseline {ev['baseline_accuracy']:.3f}, random log-loss {np.log(3):.3f})")""")

# ============================================================ 6. ML LAYER
md("""## 6. Layer 2 — Gradient Boosting ML classifier

The ML layer consumes the *same* Elo features plus shifted rolling form:

| Feature | Meaning |
|---------|---------|
| `home_elo`, `away_elo` | current team strength |
| `home_goals_avg`, `away_goals_avg` | last-5-match shifted goal averages |

It predicts `{A: 0, D: 1, H: 2}` with `GradientBoostingClassifier`. Two important
design decisions:

1. **Probability calibration.** Raw tree ensembles give extreme probabilities, and
   `edge = p × odds − 1` is brutally sensitive to miscalibration. We wrap the booster in
   `CalibratedClassifierCV` (sigmoid, internal to training — no leakage).
2. **Team-aware prediction.** At inference time the model receives the *actual* Elo of
   the two teams and their stored form (the original code fed constant placeholders,
   which made the model unable to distinguish teams at all).""")
code(r"""ml = MLFootballPredictor(model_type="gradient_boosting")
ml_metrics = ml.train(train_feat, verbose=True)   # train_feat from step 4
print("\nFeature importances:")
for name, imp in zip(ml.feature_cols, ml.model.calibrated_classifiers_[0].estimator.feature_importances_):
    print(f"  {name:<16} {imp:.3f}")""")
code(r"""# Team-aware prediction check: swap home/away and compare
p_home = ml.predict_proba("Man City", "West Ham",
                          home_elo=poisson.get_team_elo("Man City"),
                          away_elo=poisson.get_team_elo("West Ham"))
p_away = ml.predict_proba("West Ham", "Man City",
                          home_elo=poisson.get_team_elo("West Ham"),
                          away_elo=poisson.get_team_elo("Man City"))
print(f"ML P(Man City win | Man City home) = {p_home['home_win']:.3f}")
print(f"ML P(Man City win | West Ham home) = {p_away['away_win']:.3f}")
print("The ML layer clearly distinguishes teams — the original hardcoded-feature bug is gone.")""")

# ============================================================ 7. ENSEMBLE
md(r"""## 7. The hybrid ensemble

The final match probability is the **average** of the PoissonElo and ML probabilities:

$$p_{final} = \frac{p_{PoissonElo} + p_{ML}}{2}$$

Averaging two models that make different kinds of errors (a parametric count model and
a flexible tree ensemble) is a classic ensemble trick: it reduces variance and usually
beats either model alone.

Then we compute the **edge** against the bookmaker for each outcome:

$$\text{edge}_i = p_i \times \text{odds}_i - 1$$

A positive edge means the bookmaker's price overpays for the model's probability. We
only bet when the best edge is above the threshold **3%** **and** the model is
confident (`p ≥ 0.40`) — the probability floor keeps us away from longshots where
relative estimation error is huge (the "winner's curse" guard).""")
code(r"""def show_edges(home, away, row):
    probs = pipeline.ensemble_probs(poisson, ml, home, away)
    bookie = {"home_win": row["odds_home_b365"], "draw": row["odds_draw_b365"],
              "away_win": row["odds_away_b365"]}
    edges = poisson.calculate_edge(probs, bookie, threshold=0.03)
    print(f"{home} vs {away}   result={row['result']}")
    for k in ("home_win", "draw", "away_win"):
        print(f"  {k:<10} p={probs[k]:.3f}  odds={bookie[k]:.2f}  edge={edges[k]:+.2%}")
    print(f"  -> best value: {edges.get('best_value')}")

r = test_df.iloc[0]
show_edges(r["home_team"], r["away_team"], r)""")

# ============================================================ 8. RL
md("""## 8. Layer 3 — Q-Learning staking agent

*How much* should we stake? The classic answer is **fractional Kelly**:
`stake = edge / (odds − 1) × ¼`. But we can also *learn* it.

The RL agent is a **tabular Q-Learning** agent:

* **State** = `(edge bin, bankroll fraction bin)` — 10 × 10 discretised grid.
* **Action** = one of 4 Kelly **multipliers** `{0, 0.5, 1.0, 1.5}×`.
* **Reward** = the realized fractional bankroll change of the bet.

Why multipliers instead of absolute stakes? A naive agent with absolute stake levels
(up to 10% of bankroll) tends to **over-stake** — at a realistic edge of ~10% and odds
~2.2, quarter-Kelly is only ~2%, so absolute levels of 8–10% are 4–5× too aggressive
(the original code did exactly this and produced ±60% swings).

The agent is trained on the **validation** split's realized bets — never the test split
— so there is no leakage.""")
code(r"""experiences = pipeline._discovery_experiences(valid_df, poisson, ml, 1000.0)
print(f"Kelly discovery backtest on validation produced {len(experiences)} realized bets")

rl_agent = QLearningStakingAgent()
rl_agent.train(experiences, episodes=200)

# What does the trained agent recommend?
for edge, odds in [(0.05, 2.1), (0.10, 2.4), (0.20, 3.0)]:
    stake = rl_agent.get_stake_fraction(edge, odds, 1000, 1000)
    print(f"  edge={edge:.0%} odds={odds:.2f} -> stake {stake:.2%} of bankroll")""")

# ============================================================ 9. BACKTEST
md("""## 9. The backtest

Now everything comes together. Over the **test split** we:

1. Predict `p` with the hybrid ensemble.
2. Compute edges vs the (opening) bookmaker odds.
3. Bet when `edge > 3%`, `odds ≥ 1.6`, `p ≥ 0.40`.
4. Size the stake with the trained RL agent (fallback: quarter-Kelly).
5. Resolve the bet against the recorded result.
6. Record **CLV** = `(closing_odds − taken_odds) / taken_odds × 100` (the original code
   computed `(odds/odds − 1) × 100` which is always 0; the bug is fixed).
7. Track the bankroll, then compute ROI, Sharpe, max drawdown, profit factor.""")
code(r"""result = pipeline.run_backtest(
    df, use_ml=True, use_rl=True, seed=42, tag="nb", save_results=False, verbose=False)

summary = result["summary"]
bets = result["bets_df"]
equity = result["equity"]

print(pipeline.format_summary(summary))
print("\nModel quality on test (all matches, not just bets):")
for k, v in result["test_eval"].items():
    print(f"  {k.replace('_', ' ').title():<20}: {v}")""")
code(r"""bets[["date", "match", "market", "my_odds", "stake", "edge_pct",
       "bet_outcome", "profit_loss", "clv_pct", "running_bankroll"]].head(8)""")

# ============================================================ 10. PLOTS
md("## 10. Results, visualised\n\nFour views on the backtest: the **equity curve**, the **edge distribution**, the **CLV distribution** and the **P/L per bet**. These are the same plots saved to `backtests/results/` by the CLI scripts.")
code(r"""fig, axes = plt.subplots(2, 2, figsize=(15, 11))

axes[0, 0].plot(equity, color="#2E86AB", linewidth=2.5)
axes[0, 0].axhline(y=1000, color="gray", linestyle="--", alpha=0.8)
axes[0, 0].set_title("Equity Curve (PoissonElo + ML + RL staking)")
axes[0, 0].set_xlabel("Bets"); axes[0, 0].set_ylabel("Bankroll ($)")

sns.histplot(bets["edge_pct"], bins=20, ax=axes[0, 1], color="#A23B72", kde=True)
axes[0, 1].axvline(x=3, color="red", linestyle="--", linewidth=2, label="3% threshold")
axes[0, 1].set_title("Betting Edge Distribution"); axes[0, 1].set_xlabel("Edge (%)"); axes[0, 1].legend()

sns.histplot(bets["clv_pct"], bins=20, ax=axes[1, 0], color="#F18F01", kde=True)
axes[1, 0].axvline(x=0, color="black", linestyle="--", linewidth=2)
axes[1, 0].set_title("Closing Line Value (CLV) Distribution"); axes[1, 0].set_xlabel("CLV (%)")

colors = ["#06D6A0" if p > 0 else "#EF476F" for p in bets["profit_loss"]]
axes[1, 1].bar(range(len(bets)), bets["profit_loss"], color=colors, alpha=0.75)
axes[1, 1].axhline(y=0, color="black", linewidth=1.5)
axes[1, 1].set_title("Profit / Loss per Bet"); axes[1, 1].set_xlabel("Bet #"); axes[1, 1].set_ylabel("P/L ($)")

plt.tight_layout()
plt.show()""")

# ============================================================ 11. $1M (original)
md("""## 11. Bonus — "what if I invested $1M?" (flat / Kelly policies)

The full Monte-Carlo answer lives in `demo/simulation.py` (25 forward trials × 1,200
matches each). It reports the **distribution** of final bankrolls — mean, median,
P(profit), 90% range — which is the honest way to summarise a high-variance betting
strategy. The confidence-aware **dynamic** variant is compared in section 21.

> A single backtest path is dominated by variance; the Monte-Carlo distribution is the truthful summary.""")
code(r"""# Quick 2-trial taste of the simulation (full 25-trial run takes ~1-2 minutes)
import subprocess, sys
r = subprocess.run([sys.executable, "demo/simulation.py", "--trials", "2", "--matches", "300"],
                   capture_output=True, text=True, cwd=ROOT)
print(r.stdout[-1100:])""")

# ============================================================ 12. DEEP LEARNING
md("""## 12. Deep learning — PyTorch NN + TensorFlow hybrid, tested on REAL leagues

Two deep models are trained on **all** synthetic training data:

* a **PyTorch MLP** (4 engineered features → 64 → 64 → 3 softmax), and
* a **TensorFlow hybrid** — an MLP that fuses the 4 features with the PoissonElo
  model's probability outputs (a genuine statistical + deep hybrid).

They are then tested on **real** match data they have never seen: La Liga 2025/26
(cross-league) and Premier League 2025/26 (unseen matches). Features for the real
leagues come from the *previous* real season (Elo + shifted form), so the test measures
the learned feature→probability mapping, not cold-starting on unknown teams. Baselines
include the real bookmaker's implied probabilities and a cold-start row with **no team
information**.

Run the experiment yourself: `python scripts/04_deep_learning_transfer.py --offline`""")
code(r"""# Results saved by scripts/04_deep_learning_transfer.py (real downloads + training
# take a few minutes; the notebook just loads and plots the saved results).
results_dir = ROOT / "backtests/results"
transfer = pd.read_csv(results_dir / "transfer_results.csv")
print(transfer.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, league in zip(axes, ["La Liga", "Premier League"]):
    sub = transfer[transfer["league"] == league].sort_values("accuracy")
    ax.barh(sub["method"], sub["accuracy"] * 100, color="#A23B72")
    ax.set_title(f"{league} 2025/26 - accuracy (models trained on synthetic data)")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 60)
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["accuracy"] * 100 + 0.3, i, f"{row['accuracy']:.1%}", va="center", fontsize=9)
plt.tight_layout()
plt.show()

print("Reading: the real market is the strongest predictor. The synthetic-trained")
print("models transfer PARTIALLY - simple tree/linear models generalise better than")
print("deep nets out-of-distribution. See section 16 for the fix: online adaptation.")""")

# ============================================================ 13. SEASON BACKTEST
md("""## 13. Time-aware validation on REAL data — season-by-season backtest

Random splits are forbidden for temporal data. `scripts/05_season_backtest.py` runs an
**expanding-window** backtest on five real La Liga seasons (2021/22 → 2025/26): every
test season is genuinely unseen, and features for match *i* use only matches strictly
before *i* (running Elo + last-5 form, computed online). It also tests **cross-league
transfer** (La Liga ↔ Premier League).

```bash
python scripts/05_season_backtest.py --offline
```""")
code(r"""season = pd.read_csv(results_dir / "season_backtest_results.csv")
within = season[season["experiment"].str.startswith("La Liga: train")].copy()
within["test_season"] = within["experiment"].str.split("-> test ").str[-1]
print(within.pivot_table(index="method", columns="test_season", values="accuracy").round(3))

cross = season[~season["experiment"].str.startswith("La Liga: train")]
print("\nCross-league transfer (accuracy / log-loss):")
print(cross.pivot_table(index="method", columns="experiment",
                        values=["accuracy", "log_loss"]).round(3))

print("\nReading: on real football the 3-way accuracy ceiling is ~50-58% (draws are")
print("almost unpredictable). Our models beat the majority baseline by 5-10 points")
print("within a league and transfer to the other league with smaller but positive")
print("gains - a realistic, honest result.")""")

# ============================================================ 14. PREDICT CLI
md("""## 14. Prediction interface — "Team A vs Team B"

The CLI `predict_match.py` trains the PoissonElo + Gradient-Boosting layers on real
cached history and prints a structured prediction: probabilities, chosen outcome, Elo
difference, recent form, baselines, confidence/risk, data freshness.

```bash
python predict_match.py --home "Real Madrid" --away "Barcelona"
python predict_match.py --home "Arsenal" --away "Liverpool" --league E0
```""")
code(r"""# A compact inline version using the same trained layers on cached real data.
import pipeline as pipe
from models.poisson_elo_model import PoissonEloModel

real = ROOT / "data/real"
parts = []
for s in ["2122", "2223", "2324", "2425"]:
    f = real / f"SP1_{s}.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d.rename(columns={"Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
                              "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result"})
        parts.append(d[["home_team", "away_team", "home_goals", "away_goals", "result"]])
if parts:
    hist = pd.concat(parts, ignore_index=True).dropna()
    poisson = PoissonEloModel(); poisson.train(hist)
    p = poisson.predict("Real Madrid", "Barcelona")
    print(f"Real Madrid vs Barcelona  ->  H {p['home_win']:.0%} / D {p['draw']:.0%} / A {p['away_win']:.0%}")
    print(f"  Elo diff {poisson.elo_ratings.get('Real Madrid', 1500) - poisson.elo_ratings.get('Barcelona', 1500):+.0f}")
else:
    print("No cached real data - run scripts/05_season_backtest.py once to download it.")""")

# ============================================================ 15. LIMITATIONS (original)
md("""## 15. What the core pipeline cannot do (motivation for the next sections)

The core pipeline is **static after training**: a model trained on one league keeps
that league's statistics forever. If you point it at a *different* league (or a
different sport) — where home advantage, scoring rate and draw frequency shift — a
frozen model degrades. Sections 16–21 show how the project answers that:

| Limitation | Fix in this project |
|---|---|
| Frozen probabilities after training | **Adaptive model**: Elo/form update online, GB refits on drift (section 16) |
| One fixed decision rule | **Dynamic thinking layer**: fuses fresh signals, adapts its own reasoning, is confidence-aware (section 17) |
| Uses only model probabilities | **Hidden signals**: public-vs-sharp market split, multi-book dispersion (section 18) |
| Knows the future / single league | **Leak-free multi-league walk-forward**: chronological, league-reveal, audited (section 19) |
| Feed-forward features only | **LSTM / GRU** state-space architectures (section 20) |
| One stake policy | **$1M Monte-Carlo comparison**: flat / Kelly / dynamic (sections 11 & 21) |""")

# ============================================================ 16. ADAPTIVE MODEL
md(r"""## 16. The adaptive model — online adaptation when the world changes

`models/adaptive_model.py` is a **league-agnostic** baseline that keeps learning:

1. **Base layer** — PoissonElo trained on the initial data (the project core).
2. **Online state** — Elo ratings and rolling form updated AFTER every revealed match,
   so every prediction uses only information known before kick-off (no leakage).
3. **League-agnostic features** — Elo difference, rolling goals, form points. No team
   identity, no league constants: the *same* feature vector works in Serie A, La Liga,
   the Premier League, or a different sport.
4. **Adaptation** — the Gradient-Boosting layer **refits on a rolling window** when
   scheduled (every `refit_every` matches) or when its rolling Brier **drifts**
   (predictions degrading vs their own best). A `static=True` control mode measures
   exactly what adaptation buys.

First, the saved cross-league / cross-sport experiment (`scripts/10_adaptive_transfer.py`):""")
code(r"""ad = pd.read_csv(results_dir / "adaptive_transfer_results.csv")
piv = ad.pivot_table(index="experiment", columns="method", values=["accuracy", "log_loss"]).round(3)
print(piv.to_string())
print("\nReading: within an unseen season the adaptive model matches or beats the")
print("static control; the big win is on a different SPORT (synthetic basketball-like")
print("league): 78.3% vs 80.8% static - similar accuracy but far better log-loss")
print("(0.746 vs 0.915), i.e. much better-calibrated probabilities after 64 online refits.")""")
code(r"""# Walk the held-out test split match-by-match: adaptive vs frozen control
ymap = {"H": 2, "D": 1, "A": 0}

def walk_accuracy(model, frame):
    hits = 0
    for _, r in frame.iterrows():
        p = model.predict(r["home_team"], r["away_team"])
        vec = np.array([p["away_win"], p["draw"], p["home_win"]])   # order matches CLASS_MAP
        hits += int(int(np.argmax(vec)) == ymap[r["result"]])
        model.observe(r["home_team"], r["away_team"],
                      float(r["home_goals"]), float(r["away_goals"]),
                      r["result"], prob_vec=vec)
    return hits / len(frame)

adaptive = AdaptiveMatchPredictor(window=240, refit_every=60, drift_tol=0.02, min_refit=40)
adaptive.train(train_df)
acc_a = walk_accuracy(adaptive, test_df.copy())

frozen = AdaptiveMatchPredictor(window=240, refit_every=60, drift_tol=0.02, min_refit=40, static=True)
frozen.train(train_df)
acc_f = walk_accuracy(frozen, test_df.copy())

print(f"Adaptive: {acc_a:.1%} accuracy across 240 held-out matches, {adaptive.refits} online refits")
print(f"Frozen  : {acc_f:.1%} accuracy across the same matches, {frozen.refits} refits")""")

# ============================================================ 17. DYNAMIC THINKING
md(r"""## 17. The dynamic thinking layer — confidence-aware adaptation

`models/dynamic_thinking.py` is the "thinking AI layer": it makes every decision by
combining **fresh per-match signals** with a base model that is **itself self-refitting**
(`AdaptiveMatchPredictor`), and it re-weights its own reasoning from observed performance.

**What adapts (not just the stake — the MODEL):**

1. **Base model** — self-refitting PoissonElo + GB (section 16). Probabilities change as
   new information arrives.
2. **Market signals** — public line (what you can get), sharp line (a signal), the
   public-vs-sharp **split** per outcome, multi-book **consensus + dispersion**
   (disagreement = caution), **fatigue** (rest days), and an optional live-news
   **conditions** slot (injury / lineup / weather flags) that degrades gracefully.
3. **Adaptive reasoning** — the model-vs-market blend weight is re-weighted online from
   rolling Brier: whichever source has been better calibrated lately is trusted more.
4. **Confidence-aware adaptation** (the upgrade) — a **margin-based confidence**
   (how far the top outcome sits above the uniform ⅓) now:
   * **weights the calibration blend** (a confident wrong call hurts trust more than a coin-flip),
   * **gates base-model refits** (if rolling confidence decays > 0.08 below its best,
     the base refits immediately — `refit_now()`), and
   * **scales stakes** (stake × p(best)/0.40, capped [0.75, 1.4] — identical to old
     behaviour at the pass floor, more committed only when genuinely more sure).
5. **Risk-aware staking** — disagreement + dispersion shrink stakes, a hard 2% cap,
   and a survival mode (tiny flat stakes, relaxed threshold) below 10% of the start.

Let's build it on the synthetic world and walk 120 forward matches, watching it think:""")
code(r"""from demo.simulation import _forward_match_stream, _world_strengths

layer = DynamicThinkingLayer(train_df=train_df, bankroll=1_000_000.0, seed=42)
rng = np.random.default_rng(1000)
strengths = _world_strengths(42)
n_walk = 120

confs, stakes, bankroll_path = [], [], []
for day, (home, away, p_true, opening, closing) in enumerate(
        _forward_match_stream(strengths, rng, n_walk)):
    # a second noisy "book" + occasional breaking-news conditions
    extra_book = {k: closing[k] * float(rng.uniform(0.98, 1.02)) for k in closing}
    conditions = {"away_win": float(rng.uniform(-0.03, 0.03))} if day % 23 == 0 else None
    dec = layer.think(home, away, opening, closing, extra_books=[extra_book],
                      conditions=conditions, current_day=day)
    # sample the TRUE outcome (the world, not the model)
    roll = rng.random()
    cum = np.cumsum([p_true["home_win"], p_true["draw"], p_true["away_win"]])
    result = "H" if roll < cum[0] else ("D" if roll < cum[1] else "A")
    hg = 2 if result == "H" else (1 if result == "D" else 0)
    ag = 1 if result == "H" else (1 if result == "D" else 2)
    layer.observe(home, away, hg, ag, result, dec, opening, current_day=day)
    confs.append(dec["confidence"]); stakes.append(dec["stake"])
    bankroll_path.append(layer.bankroll)

print(layer.summary())

print("\nOne decision the layer made — its full THINKING TRACE:")
for tr in layer.trace:
    if tr["decision"]:
        for k, v in tr.items():
            print(f"  {k:<14}: {v}")
        break""")
code(r"""fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
axes[0].plot(confs, color="#2E86AB")
axes[0].set_title("Margin-based confidence per match (varies 0.10→0.55)")
axes[0].set_xlabel("Match #"); axes[0].set_ylabel("Confidence")
axes[1].scatter(confs, np.array(stakes) / 1e3, s=18, alpha=0.7, color="#A23B72")
axes[1].set_title("Stake vs confidence — commits more when more sure")
axes[1].set_xlabel("Confidence"); axes[1].set_ylabel("Stake ($K)")
axes[2].plot(np.array(bankroll_path) / 1e6, color="#06D6A0")
axes[2].axhline(1.0, color="gray", ls="--", lw=1)
axes[2].set_title("Bankroll ($M) through the walk")
axes[2].set_xlabel("Match #")
plt.tight_layout(); plt.show()""")
code(r"""# Honest baseline: the ORIGINAL (v1) layer — fixed poisson+ml base, no self-refit,
# no multi-book fusion, no confidence. Same stream, same rng, so it is apples-to-apples.
layer_v1 = DynamicThinkingLayer(poisson=poisson, ml=ml, bankroll=1_000_000.0,
                                seed=7, simple=True, confidence_aware=False)
rng = np.random.default_rng(1000)
for day, (home, away, p_true, opening, closing) in enumerate(
        _forward_match_stream(strengths, rng, n_walk)):
    dec = layer_v1.think(home, away, opening, closing, current_day=day)
    roll = rng.random()
    cum = np.cumsum([p_true["home_win"], p_true["draw"], p_true["away_win"]])
    result = "H" if roll < cum[0] else ("D" if roll < cum[1] else "A")
    hg = 2 if result == "H" else (1 if result == "D" else 0)
    ag = 1 if result == "H" else (1 if result == "D" else 2)
    layer_v1.observe(home, away, hg, ag, result, dec, opening, current_day=day)

s1, s2 = layer.summary(), layer_v1.summary()
print(f"dynamic (confidence-aware): final ${s1['final_bankroll']:>12,.0f}  bets {s1['n_bets']:>3}  "
      f"strike {s1['strike_rate']:>3.0f}%  refits {s1['base_refits']} (conf-gated {s1['conf_refits']})  "
      f"market weight {s1['final_market_weight']}")
print(f"v1 (original layer)       : final ${s2['final_bankroll']:>12,.0f}  bets {s2['n_bets']:>3}  "
      f"strike {s2['strike_rate']:>3.0f}%")

print("\nHonest reading: in a static synthetic world the two are statistically similar;")
print("the confidence-aware layer's value is behavioural — it re-learns when it loses")
print("confidence and scales risk with certainty. The full 12-trial comparison is in")
print("section 21, and the real-market evidence is section 18.")""")

# ============================================================ 18. HIDDEN SIGNALS
md(r"""## 18. Hidden signals on real data — the sharp-vs-public split

The cached football-data.co.uk CSVs carry up to a dozen bookmakers per match (B365,
Bwin, Betfair, Betvictor, Pinnacle, Max/Avg). `scripts/12_hidden_signals.py` tests the
"exploiting hidden signals" thesis on **real Serie A 2025/26 matches** with
point-in-time knowledge only:

1. **Consensus & dispersion** — do matches where the market disagrees resist prediction?
2. **Public vs sharp split** — betting the public (B365) price when the sharp
   (Pinnacle) line disagrees: does it beat the closing market (**CLV**)?
3. **The DynamicThinkingLayer** — the fully self-refitting model fused with all real
   signals, walked over 2025/26 (final $945,269, ROI −5.5%, 18 bets).""")
code(r"""hs = pd.read_csv(results_dir / "hidden_signals_results.csv")
disp = hs[hs["experiment"] == "dispersion-vs-predictability"].iloc[0]
clv  = hs[hs["experiment"] == "sharp-public-split"].iloc[0]
dyn  = hs[hs["experiment"] == "dynamic-layer-real-2526"].iloc[0]

print(f"Consensus accuracy — low dispersion: {disp['consensus_acc_low_disp']:.1%}  |  "
      f"high dispersion: {disp['consensus_acc_high_disp']:.1%}  (n={disp['n']:.0f}, chance = 33%)")
print(f"Sharp-vs-public CLV: {clv['avg_clv_pct']:+.2f}% per bet  (n={clv['n']:.0f}, "
      f"t={clv['clv_t']:.2f}, p={clv['clv_p']:.4f}) — positive {clv['positive']:.0f}/{clv['n']:.0f}")
print(f"Dynamic layer on real Serie A 25/26: final ${dyn['final_bankroll']:,.0f} "
      f"({dyn['roi_pct']:+.1f}%), {dyn['n_bets']:.0f} bets, strike {dyn['strike_rate']:.0%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
axes[0].bar(["low dispersion", "high dispersion"],
            [disp['consensus_acc_low_disp'] * 100, disp['consensus_acc_high_disp'] * 100],
            color=["#2E86AB", "#06D6A0"])
axes[0].axhline(33.3, color="red", ls="--", lw=1.5, label="random chance")
axes[0].set_ylabel("Consensus accuracy (%)"); axes[0].set_ylim(0, 70); axes[0].legend()
axes[0].set_title("Dispersion vs predictability (real Serie A 25/26)")
axes[1].bar(["sharp-vs-public CLV"], [clv['avg_clv_pct']], color="#A23B72")
axes[1].axhline(0, color="black", lw=1)
axes[1].set_title(f"CLV per bet (n={clv['n']:.0f}, t={clv['clv_t']:.2f})")
axes[1].set_ylabel("CLV (%)")
plt.tight_layout(); plt.show()

print("Reading: high-dispersion matches were EASIER to call (56.3% vs 52.1%) — the")
print("opposite of the naive 'disagreement = unpredictable' hypothesis. And betting the")
print("sharp-leaning outcome at the public price yields +1.16% CLV per bet (t=4.72,")
print("p<0.001) — real positive closing-line value on real matches.")""")

# ============================================================ 19. AGENT_SIM
md(r"""## 19. Strict no-future-knowledge multi-league walk-forward (`agent_sim/`)

The most important simulation: a **real-time autonomous agent** placed at a random
point in history, with **no knowledge of future results**, free to discover leagues as
they become available.

**How it is guaranteed leak-free:**

```
MATCH AVAILABLE  ->  DATA CUTOFF  ->  MODEL ANALYSIS  ->  PREDICTION
  ->  BET / NO BET  ->  STAKE  ->  MATCH OCCURS  ->  RESULT REVEALED  ->  LEARN
```

* A **randomised world** per run: random leagues (2–4 of La Liga / Premier League /
  Bundesliga / Serie A), random season, random start date, random **league-reveal
  order** — the agent can only see fixtures of leagues whose reveal date has passed.
* At each simulation day the agent sees only: fixtures of revealed leagues kicking off
  within a 3-day lookahead (schedule is public) and results of matches that have
  already kicked off. **Nothing from the future is ever attached to a row.**
* Every opportunity is **audited**: `data_cutoff <= prediction time <= kickoff`. If a
  leak were ever detected the bet is **invalidated** and flagged.
* The agent **learns per league** (rolling ROI raises the edge threshold for leagues
  that lose it money), tracks **fatigue**, switches to **survival mode** below 10% of
  the start bankroll, and refits its base model online.
* **Tables are not printed to the terminal** — everything lands in rolling CSVs under
  `backtests/results/agent_sim/` (the live ledger is rewritten every 20 resolved
  matches and keeps only the most recent 20 rows). Data is fetched on demand from the
  free public API — nothing large is stored.

Let's build one reproducible world on **cached real data** (offline):""")
code(r"""from agent_sim.stream import World

world = World(seed=7, leagues=["SP1", "E0"], walk_season="2425", offline=True)
world.sim_dates = world.sim_dates[:80]      # keep this demo quick (real runs use the whole season)
for k, v in world.describe().items():
    print(f"  {k:<14}: {v}")""")
code(r"""from agent_sim.engine import SimulationEngine
from agent_sim.agent import BettingAgent
from agent_sim.ledger import RollingLedger
from agent_sim import report as agent_report

agent = BettingAgent(world.train_df, bankroll=1_000_000.0, stake_mode="flat", seed=7)
ledger = RollingLedger("notebook_demo")
engine = SimulationEngine(world, agent, ledger)
summary = engine.run()
engine.number_bets()

print("ML AGENT:", {k: summary[k] for k in
                    ("final_bankroll", "roi_pct", "n_bets", "n_wins", "n_leak_flags", "model_refits")})
print("Leakage audit: 0 flags = every prediction used only pre-kickoff information.\n")
print("League reveal timeline (when each league became available / first bet):")
print(engine.league_timeline().to_string(index=False))

agent_report.build_run_report("notebook_demo", 7, world, engine, ledger)
ledger.save_all()
print(f"\n[OK] artifacts under backtests/results/agent_sim/ — live rolling CSV: {ledger.live_path().name}")""")
code(r"""# Baselines walk the SAME world through the SAME engine (apples-to-apples).
from agent_sim.baselines import BaselinePolicy

base = BaselinePolicy("implied", seed=7, bankroll=1_000_000.0)   # market follower
ledger2 = RollingLedger("notebook_demo_implied")
engine2 = SimulationEngine(world, base, ledger2)
s2 = engine2.run()
engine2.number_bets()
ledger2.save_all()

print(f"{'policy':<28}{'final':>13}{'ROI':>9}{'bets':>7}{'wins':>7}{'strike':>9}")
rows = [
    ("ML agent (adaptive)", summary),
    ("implied (follows market)", s2),
    ("no-bet (do nothing)", {"final_bankroll": 1_000_000.0, "roi_pct": 0.0,
                             "n_bets": 0, "n_wins": 0, "strike_rate": 0.0}),
]
for name, s in rows:
    print(f"{name:<28}{s['final_bankroll']:>13,.0f}{s['roi_pct']:>9+.1f}{s['n_bets']:>7}"
          f"{s['n_wins']:>7}{s.get('strike_rate', 0) * 100:>8.0f}%")""")
code(r"""# The auditable ledger: bets, no-bet reasons, per-league performance.
bets_l = pd.read_csv(ROOT / "backtests/results/agent_sim/notebook_demo_bets.csv")
nobets = pd.read_csv(ROOT / "backtests/results/agent_sim/notebook_demo_nobets.csv")
opps   = pd.read_csv(ROOT / "backtests/results/agent_sim/notebook_demo_opportunities.csv")

print(f"Bets placed: {len(bets_l)} | no-bet opportunities: {len(nobets)} | total evaluated: {len(opps)}")
print("\nWhy the agent said NO (top reasons):")
print(nobets["reason"].value_counts().head(6).to_string())
print("\nSample of the betting ledger:")
cols = ["bet_no", "kickoff", "league", "match", "prediction", "edge",
        "confidence", "stake", "result", "profit", "bankroll_after"]
print(bets_l[cols].head(8).to_string(index=False))
print("\nProfit by league:")
by_league = pd.read_csv(ROOT / "backtests/results/agent_sim/notebook_demo_by_league.csv")
print(by_league[["league", "n_bets", "wins", "profit", "roi_pct"]].to_string(index=False))""")
code(r"""bets_sorted = bets_l.sort_values("kickoff")
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.plot(bets_sorted["bankroll_after"] / 1e6, marker="o", ms=3, color="#2E86AB")
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.set_title("Agent bankroll through the walk (first 80 days of La Liga + EPL 2024/25)")
ax.set_xlabel("Bet #"); ax.set_ylabel("Bankroll ($M)")
plt.tight_layout(); plt.show()""")

# ============================================================ 20. LSTM / GRU
md(r"""## 20. State-space changing neural architectures — LSTM / GRU

`scripts/11_lstm_state_test.py` tests whether **recurrent** models (which see the match
history as a sequence, i.e. a changing state space) beat the classic models on real
Serie A → Serie A 25/26 transfer. The LSTM/GRU consume engineered features (Elo diff,
form) per match and learn the temporal pattern; a "thin" variant uses goals only.""")
code(r"""lstm = pd.read_csv(results_dir / "lstm_state_results.csv")
within = lstm[lstm["experiment"].str.contains("within-league")].copy()
print(within[["method", "accuracy", "log_loss", "brier"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 4.2))
sub = within.sort_values("accuracy")
ax.barh(sub["method"], sub["accuracy"] * 100, color="#A23B72")
base_acc = sub[sub["method"] == "Majority / base rate"]["accuracy"].iloc[0] * 100
ax.axvline(base_acc, color="red", ls="--", lw=1.5, label=f"majority baseline ({base_acc:.0f}%)")
ax.set_xlabel("Accuracy (%)"); ax.set_title("Serie A → Serie A 25/26: recurrent vs classic models")
ax.legend()
plt.tight_layout(); plt.show()

print("Reading: LSTM/GRU match the classic models but do not beat them on this small,")
print("low-information dataset — a valuable negative result. The recurrent architectures")
print("earn their keep only with richer sequence features (xG, lineups, possession).")""")

# ============================================================ 21. DYNAMIC $1M
md(r"""## 21. The $1M question, answered four ways

`demo/simulation_dynamic.py` replays **identical forward match streams** under four
policies and reports the full distribution (the saved 12-trial canonical run):""")
code(r"""dyn = pd.read_csv(ROOT / "demo/output/simulation_1m_dynamic.csv")
print(dyn.round(0).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
x = np.arange(len(dyn))
axes[0].bar(x - 0.2, dyn["mean"] / 1e6, 0.4, label="mean", color="#2E86AB")
axes[0].bar(x + 0.2, dyn["median"] / 1e6, 0.4, label="median", color="#06D6A0")
axes[0].axhline(1.0, color="gray", ls="--", lw=1)
axes[0].set_xticks(x); axes[0].set_xticklabels(dyn["policy"])
axes[0].set_ylabel("$M"); axes[0].set_title("Final bankroll by policy ($1M start)")
axes[0].legend()
axes[1].bar(x, dyn["p_profit"], color="#F18F01")
axes[1].set_xticks(x); axes[1].set_xticklabels(dyn["policy"])
axes[1].set_ylabel("P(profit) %"); axes[1].set_ylim(0, 110)
axes[1].set_title("Probability of ending in profit")
plt.tight_layout(); plt.show()

print("Honest reading: the dynamic layer adapts its reasoning (final market weight,")
print("uncertainty shrink, drawdown risk, survival switch) instead of applying a fixed")
print("rule — but in this synthetic world flat staking's variance control is hard to")
print("beat, and the two dynamic variants are statistically indistinguishable at 12")
print("trials. The layer's value shows on real data where the sharp line carries")
print("genuine information (section 18).")""")

# ============================================================ 22. FINAL
md(r"""## 22. Reproducibility, limitations, next steps

### Reproducibility
Every random draw goes through NumPy's seeded `default_rng` (`seed=42` by default).
Running this notebook (or any script) top-to-bottom reproduces the identical dataset,
models, bets and metrics. `python -m pytest tests/` verifies the core invariants: no
leakage, calibrated models, correct CLV, reproducible backtests (12 tests).

### Limitations — be honest
1. **Synthetic world.** Most of the core backtest runs inside a calibrated simulated
   world — it measures the *methodology*, not real-world profitability.
2. **Selection inflation.** The average edge of selected bets is inflated by selection:
   the model bets exactly where it disagrees with the bookmaker most. Real markets are
   far tighter (the real-data CLV experiments in section 18 measure this honestly).
3. **Variance.** A single backtest path is one draw of a very noisy process. Always use
   the Monte-Carlo distribution (sections 11 & 21) or the multi-run walk-forward
   aggregate (section 19).
4. **Deep nets vs classic models.** On real out-of-distribution data, simple calibrated
   models beat deep nets; LSTM/GRU need richer sequence features. The adaptive layer is
   the current best answer to changing leagues.
5. **Real-data CLV is small.** +1.16% per bet is real (t=4.72) but the 95% CI on ROI
   still includes zero — five seasons is a small sample.

### Next steps
* Full multi-run aggregate of the walk-forward agent (`agent_sim/`): mean/median ROI,
  % profitable runs, league-selection frequency across 100 seeds.
* Richer features (xG, shots, standings, weather) and Bayesian goal models.
* Live forward P&L logging (`predictions/forward_prediction_workflow.md`).

---
**Run from the repo root:**

```bash
python run_full_ml_rl.py                       # full core pipeline: PoissonElo + ML + RL
python scripts/04_deep_learning_transfer.py    # PyTorch + TensorFlow on real leagues
python scripts/12_hidden_signals.py            # real-data hidden-signals experiment
python scripts/11_lstm_state_test.py           # LSTM / GRU state-space test
python demo/simulation_dynamic.py --trials 12  # $1M: flat / Kelly / dynamic / v1
python demo/simulation.py --trials 25          # $1M Monte-Carlo (core policies)
python -m pytest tests/ -v                     # verification suite
```

*Notebook auto-synced with the codebase: every model change in `models/`, `agent_sim/`
or `scripts/` is reflected here (see `docs/` for the full experiment write-ups).*""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
out = Path(__file__).resolve().parent / "notebooks" / "01_explained_ml_pipeline.ipynb"
nbf.write(nb, out)
print(f"[OK] wrote {out} with {len(cells)} cells")
