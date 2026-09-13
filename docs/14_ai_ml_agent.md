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

## 4. Head-to-head results (same model, same data, same season — only the AI layer differs)

Unit ROI = profit / staked — the fair metric, because downsizing changes bet counts.

| Run | ML-only unit ROI | AI+ML unit ROI | AI delta |
|---|---|---|---|
| I1, 4 seasons (2223–2526) | -15.51% (48 bets) | -7.41% (13 bets) | **+8.09 pts** |
| E0, 3 seasons (2324–2526) | +17.47% (50 bets) | +43.39% (13 bets) | **+25.92 pts** |
| SP1, 3 seasons (2324–2526) | -19.26% (31 bets) | -17.15% (13 bets) | **+2.11 pts** |
| I1, 2 seasons (2425–2526) | -100.0% (12 bets) | +3.26% (11 bets) | **+103.26 pts** |
| **Pooled (10 league-seasons)** | **≈ -3.7%** | **≈ +3.8%** | **≈ +7.5 pts** |

Key properties:

- **Positive pooled unit ROI for the hybrid; negative for ML-only** — on 3 different
  leagues, 10 league-seasons, leakage-free training (5 prior seasons only).
- The AI layer never saw a result in advance; every gate uses pre-kickoff odds.
- It works with **less data**: training is 1 league × 5 seasons, and results hold
  with as few as 2 prior seasons (`ai_ml_headtohead_train2.csv`).
- Caveat: with ~13 bets/season the sample is small — this is a research harness, not
  a trading claim. Next step would be per-bet CLV measurement (the honest metric that
  converges fastest) and a bootstrap over bet order.

## 5. Run it

```bash
.venv/Scripts/python.exe demo/simulation_ai_ml.py --offline                 # I1 2526
.venv/Scripts/python.exe demo/simulation_ai_ml.py --offline --league E0 \
    --test-seasons 2324 2425 2526
.venv/Scripts/python.exe agent_sim/ai_agent.py                              # smoke test
```
