# Research → Engineering Decision Table

## Source 1: DrawBias Quantitative Model Guide

| Research Finding | Required? | Existing Implementation | Gap | Action | Evidence |
| --- | --- | --- | --- | --- | --- |
| Data → Features → Statistical/ML → Probability → Market comparison | YES | pipeline.py + models/ | Partial: no de-vig in betting loop | Verify pipeline covers full chain | DrawBias: "core architecture" |
| Out-of-sample backtesting | YES | agent_sim/engine.py has chronological walk-forward | Exists | Verify no random splits in main eval | DrawBias: "performance on training data alone proves very little" |
| Margin removed before comparing to market | YES | models/calibration.py: implied_probs() | EXISTS but not used in all backtests | Ensure all bet loops use de-vigged probs | DrawBias: "margin must be removed before comparing" |
| Version control for model changes | PARTIAL | analysis/experiment_tracker.py | No model version per prediction | Add model_version to every bet record | DrawBias: "track exactly what changed and when" |
| Live results compared to backtest | NO | No live/paper separation | Missing | Build paper trading with snapshot | DrawBias: "ongoing monitoring matters" |
| Edge decay monitoring | NO | Not implemented | Missing | Add rolling CLV and edge decay tracking | DrawBias: "markets evolve, sportsbooks improve" |
| Human review of situational factors | YES | analysis/research_layer.py | Exists but not connected to main pipeline | Connect research layer to backtest | DrawBias: "model + human review outperforms" |
| Ensemble of diverse models | YES | models/stacking_ensemble.py | Exists but not tested on real data | Test stacking on real data | DrawBias: "combine genuinely diverse approaches" |

## Source 2: quantbet (GitHub)

| Research Finding | Required? | Existing Implementation | Gap | Action | Evidence |
| --- | --- | --- | --- | --- | --- |
| Calibration > accuracy (first-class) | YES | calibration.py exists, ml_layer uses CalibratedClassifierCV | Calibration not measured in main backtest loop | Add calibration metrics to every backtest | quantbet: "probability calibration — not accuracy — is the objective" |
| Temperature scaling | YES | Only sigmoid/isotonic in CalibratedClassifierCV | No temperature scaling | Add temperature scaling option | quantbet: "Post-hoc temperature scaling (Guo et al., 2017)" |
| Chronological train/val/test splits | YES | agent_sim/engine.py has this | Exists | Verify in all scripts | quantbet: "never random" |
| Feature scaler fit on training only | YES | ml_layer.py uses CalibratedClassifierCV with TimeSeriesSplit | Exists | Verify no scaler leak | quantbet: "Rolling features built strictly from prior matches" |
| Early stopping on validation log-loss | PARTIAL | GB has early stopping via n_estimators | Not explicit log-loss stopping | Add log-loss early stopping | quantbet: "Early stopping on validation log-loss, not accuracy" |
| Feature snapshot at bet time | NO | Not implemented | Missing | Store features at prediction time | quantbet: "bets store a feature snapshot at bet time" |
| De-overrounded market evaluation | YES | implied_probs() exists | Not used in all evaluations | Ensure all comparisons use de-vigged | quantbet: "Evaluation against de-overrounded market probabilities" |
| Feature ablation | NO | Not systematically done | Missing | Build feature ablation experiments | quantbet: "feature-ablation item sits on the roadmap" |
| Documented failures | YES | research/hypotheses.jsonl has REJECTED entries | Exists | Continue documenting | quantbet: "knowing precisely why you lose is the research skill" |
| Experience replay buffer for retraining | NO | Not implemented | Missing (low priority) | Not implementing (synthetic data) | quantbet: retrain.py |

## Source 3: Academic Literature (from RESEARCH_REVIEW.md)

| Research Finding | Required? | Existing Implementation | Gap | Action | Evidence |
| --- | --- | --- | --- | --- | --- |
| Brier score decomposition | YES | brier_score() exists in calibration.py | No decomposition into reliability/resolution | Add decomposition | Murphy (1973) |
| Calibration by odds range | NO | Not implemented | Not critical for synthetic data | Mark N/A for now | Constantinou & Fenton (2013) |
| Purged cross-validation with embargo | YES | agent_sim has embargo concept | Exists in frozen_judge.py | Verify correct implementation | De Prado (2018) |
| Deflated Sharpe Ratio | YES | scripts/16_bootstrap_validation.py has bootstrap | No DSR | Add DSR calculation | Bailey & De Prado (2012) |
| Closing Line Value as primary metric | YES | CLV tracked in pipeline.py | Exists | Make CLV first-class in all backtests | Mints (2021) |
| Fractional Kelly with caps | YES | pipeline._fractional_kelly() | Exists | Verify correct caps | Kelly (1956), Thorp (2006) |
| Walk-forward expanding windows | YES | scripts/16_bootstrap_validation.py | Exists | Verify multi-league | Arlot & Celisse (2010) |

## Summary: What to Implement

### MUST IMPLEMENT (Critical Gaps)
1. **Temperature scaling** for calibration (quantbet lesson #1)
2. **Feature snapshot at bet time** (quantbet lesson #2)
3. **Calibration metrics in main backtest loop** (quantbet lesson #1)
4. **Model version per prediction** (DrawBias version control)
5. **Automated leakage tests** (quantbet lesson #3)
6. **Proper backtest record format** with all required fields

### ALREADY EXISTS (Verify Only)
1. De-vigging via implied_probs()
2. Chronological walk-forward
3. CLV tracking
4. Isotonic calibration
5. Fractional Kelly
6. Experiment tracking
7. Research layer

### DEFER (Not needed for current stage)
1. Temperature scaling on real data (need real calibration data)
2. Experience replay buffer (synthetic data)
3. Browser/phone tracker UI (existing dashboard)
4. Paper/live separation (not ready for live)
5. Edge decay monitoring (need live data)
