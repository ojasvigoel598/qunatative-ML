# 14 — AI + ML Agent: what limits ROI, what works small-scale, and the hybrid agent

Date: 2026-09-13 · Code: `agent_sim/ai_agent.py`, `demo/simulation_ai_ml.py`
Results: `backtests/results/ai_ml_headtohead*.csv`

---

## 1. What limits the project's ROI (root causes found in this repo's own results)

| Evidence | Finding |
|---|---|
| `why_model_losing.txt` | The **winner's curse** dominates: advertised edge +14% collapses to ~+3% realised. Any bet selected because the ML's edge looks big is selected exactly where the ML is most wrong. |
| `clv_threshold_results.csv` | Raising the ML **edge threshold** makes ROI *worse* (0% → -42.4%, 4% → -45.5%). Edge-selection is the problem, not the threshold. |
| `data_size_sweep_report.md` | On small training data the failures **concentrate in longshots** (high-odds bets), not favourites. |
| `multi_run_aggregate.csv` | The pure-ML agent: mean ROI -16.9%, 0/3 runs profitable. |
| walk-forward runs (this doc) | ML-only unit ROI across I1/E0/SP1, 10 league-seasons: **-3.7%**; its longshot bets lost ~100% of stake. |

## 2. What achieved positive ROI in this repo — and how it replicated at small scale

`clv_focused_results.csv` is the only large positive-ROI result: **+53.2% (La Liga),
+14.8% (Serie A), one season each**, built on *one league · one season · ~50 bets*:

1. Bet only where **our price beats the sharp (Pinnacle) price** — expected CLV > 0.
2. Average odds ≈ 2.1 — **no longshots**.
3. `hidden_signals_results.csv` — the sharp-vs-public split carries real information
   (+1.16% CLV per bet, t=4.72, p<0.001, 94/200 positive) and is computable from
   point-in-time odds alone, i.e. it needs **less data, not more**.

That is the small-scale recipe: a market-structure filter, not a bigger model.

## 3. The AI + ML hybrid (`agent_sim/ai_agent.py`)

`AIMLBettingAgent` subclasses the ML `BettingAgent` — the model is untouched. The AI
layer reviews every ML bet using only pre-kickoff information and can VETO, DOWNSIZE
or UPSCALE it:

| Signal | Rule | Source of truth |
|---|---|---|
| Longshot guard | veto odds ≥ 3.5 | data-size sweep: small-data losses concentrate there |
| Sharp split | veto if sharp line moves against the pick | hidden signals t=4.72 |
| Dispersion | veto when books disagree > 12% | no reliable price exists |
| **Expected CLV gate** | **our price must beat the sharp line (CLV ≥ 0)** | clv_focused_results: the only proven positive-ROI filter |
| Winner's-curse sizing | shrink stakes as ML-vs-market disagreement grows | why_model_losing.txt |
| Agreement upscale | ≤ 25% bigger stake when sharp + CLV agree | capped, never a longshot |

## 4. Head-to-head results — DOUBLE-VERIFIED (multi-seed + bootstrap CI)

Unit ROI = profit / staked — the fair metric, because downsizing changes bet counts.
Each cell pools 2 test seasons (2024/25, 2025/26) × 5 training seeds, leakage-free
(training = 5 prior seasons only).  95% CI from 2,000 bootstrap resamples of the bet
ledger.  An earlier single-seed version of this table (+3.26% on I1) did **not**
survive the multi-seed check — it was seed luck; see `docs/15_verification_report.md`.

| League | ML-only unit ROI (bets, strike) | AI+ML unit ROI (bets, strike) | AI delta |
|---|---|---|---|
| I1 (Serie A) | **-100.0%** (61 bets, **0 wins**) | -24.5% (37 bets, 35.1%) | **+75.5 pts** |
| E0 (Premier) | +20.8% (164 bets, 45.1%) | **+51.0%** (63 bets, 60.3%) — CI [+13.0, +90.3] ✓ | **+30.2 pts** |
| SP1 (La Liga) | -16.6% (82 bets, 35.4%) | -14.7% (66 bets, 42.4%) | **+1.9 pts** |
| **Pooled, 476 bets** | **-14.2%** (307 bets, 33.6%) | **+4.7%** (166 bets, 47.6%) | **+18.9 pts** |

Key properties:

- **Pooled positive unit ROI for the hybrid, negative for ML-only** on 3 leagues,
  476 bets — and the E0 result is CI-significant.  I1/SP1 alone are not significant:
  small samples stay small.
- The AI layer never saw a result in advance; every gate uses pre-kickoff odds.
- Data-size sweep (calibrated gates): the hybrid needs **≥ 5 prior training seasons**;
  with 2 seasons both agents lose (`ai_ml_headtohead_train2_cal.csv`).  Small data is
  a real constraint — the AI layer filters bad bets, it cannot conjure signal that
  the model never learned.
- Honest caveat: the AI layer halves staking volume, so bankroll growth is slower
  than unit ROI suggests; and ~166 bets is still not a trading-grade sample.

## 5. Run it

```bash
.venv/Scripts/python.exe demo/simulation_ai_ml.py --offline                 # I1 2526
.venv/Scripts/python.exe demo/simulation_ai_ml.py --offline --league E0 \
    --test-seasons 2324 2425 2526
.venv/Scripts/python.exe agent_sim/ai_agent.py                              # smoke test
```
