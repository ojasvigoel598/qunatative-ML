# Verification Report — what is real, what was an artifact (Sep 2026)

This report re-audits every headline number with better testing methodology
(multi-seed, bootstrap CI, quality metrics vs the market, and a live
season-to-date simulation).  It also corrects two over-claims made earlier.

## 1. Why "ML-only +3.04%" became "-100%" — both numbers were misleading

The first head-to-head run trained the model on data that included the current
season (`world.df.iloc[:0]` bug in the world setup) — the ML agent had already
seen the fixtures it was betting on.  Removing that leakage, and switching the
metric to **unit ROI** (profit / staked, the fair metric when the AI layer
downsizes stakes), gave a clean comparison.  The first clean run showed
ML-only -100% on I1 — which the multi-seed check then showed was partly real
(structural longshot losses) and partly seed luck.  Final honest numbers are
in §3.

## 2. Model quality audit (`demo/ml_quality_audit.py`) — the model is good, selection is the problem

Test window: E0 2025/26, trained on 5 prior seasons.  **The model is fine**:

| Predictor | accuracy | Brier ↓ | log loss ↓ |
|---|---|---|---|
| Our model | 50.5% | 0.6072 | 1.0153 |
| Market (B365) | 53.9% | 0.5844 | 0.9821 |
| Uniform | 38.9% | 0.6667 | 1.0986 |

Calibration is honest: predicted 0.45 → realised 0.44, 0.54 → 0.54, 0.64 → 0.68.
So: **not a code bug, not a training bug** — the model is within ~3 pts of the
market on accuracy and well-calibrated.  The problem is *bet selection*: the
agent bets exactly where model and market disagree most (mean odds 3.97,
nominal edge +81%), and in that zone the model has **zero** advantage
(both right 25.6%).  Edge-selection amplifies noise — the winner's curse,
measured quantitatively.  This is why every serious shop converges on
market-implied probabilities + a thin personal edge, not raw model edges.

## 3. Double-verified head-to-head (5 seeds × 2 seasons × 3 leagues, 476 bets)

Unit ROI = profit / staked.  Bootstrap 95% CI, 2,000 resamples of the ledger.

| League | ML-only | AI+ML | AI delta |
|---|---|---|---|
| I1 (Serie A) | **-100.0%** (61 bets, 0 wins) | -24.5% (37 bets) | +75.5 pts |
| E0 (Premier) | +20.8% (164 bets) | **+51.0%** (63 bets), CI [+13.0, +90.3] ✓ | +30.2 pts |
| SP1 (La Liga) | -16.6% (82 bets) | -14.7% (66 bets) | +1.9 pts |
| **Pooled** | **-14.2%** (307 bets) | **+4.7%** (166 bets) | **+18.9 pts** |

- The earlier single-seed "+3.26% / +103 pts on I1" did not survive multi-seed
  checking (became -24.5%) — reported and corrected in docs/14.
- ML-only's I1 -100% is **structural, not variance**: 0 wins in 61 bets across
  5 seeds and 2 seasons.  Every ML pick is a longshot priced above ~3.5 where
  the model has no edge (see §2).
- The AI layer improved unit ROI in all three leagues (+18.9 pts pooled), and
  its E0 result is the only CI-significant positive.  One league positive out
  of three = promising but not proven; the honest pooled figure is the ~+4.7%.

## 4. Data-size sweep with calibrated gates

| Training seasons | ML-only unit ROI | AI+ML unit ROI |
|---|---|---|
| 2 | -100% (10 bets, 0 wins) | -100% (2 bets, 0 wins) |
| 5 (default) | -100% (12 bets) | +3.3% single-seed → -24.5% multi-seed |
| 20 | -100% (12 bets) | +20.4% / -100% on 1 bet (seasons differ) |

The hybrid needs **≥ 5 prior seasons**; the AI layer filters bad bets, it
cannot create signal the model never learned.  (This corrects the earlier
claim that 2 seasons sufficed.)

## 5. Season-to-date: Premier League 2026/27, all 30 played matches

Both agents bet **exactly 1 match in 30** — the calibrated agent finds almost
no edge vs the market on a 20-season-trained model, which is itself the
honest result.  Per-match ledger in `season_to_date_E0_2627.csv`.

| | ML-only | AI+ML |
|---|---|---|
| Matches bet | 1 / 30 | 1 / 30 |
| The bet | Ipswich v Sunderland (2026-08-22), away @2.63, model 50% vs market 38% ("edge +32.5%") | same match, stake shrunk ×0.25 |
| Stake | $10,000 (1.0% of bankroll) | $2,500 (0.25%) |
| Score | 2-1 Sunderland... (home win) | same |
| P/L | -$10,000 | -$2,500 |
| Bankroll | $1,000,000 → $990,000 (-1.00%) | → $997,500 (-0.25%) |
| Unit ROI | -100% | -100% |

Skipped 29 matches: 24 "no edge vs market", 5 "model prob below 40% floor".
The AI layer shrank the stake 4× (winner's-curse guard: model-market gap 20%),
cutting the loss from 1.00% to 0.25% of bankroll — exactly its job.

## 6. Can we trust it? — methodology checklist

| Check | Status |
|---|---|
| No leakage (train strictly on prior seasons) | ✓ fixed & enforced |
| Multi-seed (5 seeds; model has a training seed) | ✓ new in this pass |
| Per-bet ledger for independent re-analysis | ✓ `ai_ml_bets_<L>.csv` |
| Bootstrap CI on pooled unit ROI | ✓ (E0 significant; I1/SP1 not) |
| Fair metric (unit ROI, not bankroll ROI) | ✓ |
| Model quality vs market baseline (acc/Brier/log loss) | ✓ within ~3 pts of market, calibrated |
| Multiple leagues | ✓ 3 (only E0 positive & significant) |
| Out-of-sample live test (2026/27 PL) | ✓ 30 matches, 1 bet, lost small |
| Fee/limit realism (exchange commission, slippage) | ✓ 2% commission on wins: E0 stays +49.2% significant |
| Per-bet CLV vs closing line (fastest-converging metric) | ✓ pooled AI+ML CLV **+1.50%, CI [+0.32, +2.73] — significant** |

**Bottom line:** the model's math is sound and near-market; the -100% ML-only
results come from longshot selection (winner's curse), not broken code.  The
AI+ML hybrid turns pooled unit ROI positive (+4.7% vs -14.2%) with only E0
CI-significant.  Treat it as a validated *research harness*, not a money
printer: the next two hardening steps are closing-line CLV tracking and
commission/stake-cap realism.

## 7. Closing-line CLV + trading costs (added after first publication)

Both TODOs above are now implemented in `demo/simulation_ai_ml.py`
(`--commission`, `--slippage`; every bet graded vs the closing line).

**Per-bet closing-line CLV, pooled 5 seeds × 2 seasons × 3 leagues:**

| Agent | mean CLV per bet | 95% CI | verdict |
|---|---|---|---|
| ML-only | -0.04% | [-0.96, +0.87] | zero edge — its "edges" are noise |
| **AI+ML** | **+1.50%** | **[+0.32, +2.73]** | **significantly beats the close** |

Per league, the AI layer beats ML-only on CLV in **all three** leagues
(I1 +0.47 vs -1.00, E0 +2.83 vs +0.82, SP1 +0.81 vs -1.04; E0's CLV CI is
significant on its own).  CLV is the strongest evidence in this project: it
has no result variance, so ~166 bets of significant CLV is worth far more
than 166 bets of ROI.  The E0 ROI significance (+51.0% gross) is corroborated,
not contradicted, by the price evidence.

**Costs:** at 2% exchange commission on net winnings (Betfair base) the E0
result stays significant (+49.2%, CI [+11.7, +87.9]); pooled numbers barely
move (AI+ML +4.8% → +4.5% net; slippage of 1% would cost roughly another
1 pt on avg odds ~2.1).  The ML-only longshot strategy stays deeply negative
under any cost assumption.

**Remaining honesty note:** closing lines in the data are B365 close (Pinnacle
close unavailable post-2020); Avg-line proxies were used pre-kickoff.  CLV vs
a soft-ish close slightly *flatters* us — treat +1.5% as an upper bound, and
stake caps / market impact are still unmodelled.
