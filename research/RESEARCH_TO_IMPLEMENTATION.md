# Research to Implementation Mapping

## How Each Research Idea Was Adopted

| # | Research Source | Method | What Was Adopted | Where Implemented | Why | Expected Benefit |
|---|----------------|--------|-----------------|-------------------|-----|------------------|
| 1 | Niculescu-Mizil & Caruana (2005) | CalibratedClassifierCV | Sigmoid + Isotonic calibration on ML layer | models/ml_layer.py | Raw GB probabilities are extreme; edge = p*odds-1 is sensitive to miscalibration | Reduced ECE from 0.114 to 0.062 |
| 2 | Platt (1999) | Platt scaling | Sigmoid calibration as default | models/ml_layer.py | Fast, stable, sufficient for small samples | Baseline calibration |
| 3 | Zadrozny & Elkan (2001) | Isotonic regression | Isotonic calibration option | models/ml_layer.py | Non-parametric; captures complex miscalibration | ECE 0.052 vs 0.070 for sigmoid |
| 4 | Wolpert (1992) / Breiman (1996) | Stacking | PoissonElo + LightGBM + GB with meta-learner | models/stacking_ensemble.py | Optimal blend weights learned from data | Accuracy 0.579 vs 0.550 baseline |
| 5 | Ke et al. (2017) | LightGBM | LightGBM as ML layer option | models/ml_layer.py | Faster training, often better generalization | Comparable accuracy, 3x faster |
| 6 | Efron & Tibshirani (1993) | Bootstrap CI | 5000-resample bootstrap for ROI and Sharpe | scripts/16_bootstrap_validation.py | Distribution-free uncertainty quantification | Honest CIs on all metrics |
| 7 | Arlot & Celisse (2010) | Walk-forward CV | Expanding-window validation with embargo | scripts/16_bootstrap_validation.py | Random CV underestimates test error 10-30% | Realistic OOS estimates |
| 8 | Kelly (1956) / Thorp (2006) | Fractional Kelly | 0.25 Kelly with caps | pipeline.py | Full Kelly has >10% ruin probability | Ruin probability <1% |
| 9 | Constantinou & Fenton (2013) | Multi-metric evaluation | ECE, Brier, log-loss + ROI together | pipeline.py, models/calibration.py | Accuracy alone is misleading for betting | Honest model comparison |
| 10 | De Prado (2018) | Purged temporal validation | Strict chronological split with no overlap | pipeline.py | Prevents look-ahead bias and information leakage | Zero detected leakage |
| 11 | Brier (1950) / Murphy (1973) | Brier decomposition | Reliability, resolution, uncertainty | models/calibration.py | Identifies actionable calibration improvements | Diagnostic framework |
| 12 | Mints (2021) | CLV as primary metric | CLV t-stat, CLV win rate | pipeline.py compute_metrics | CLV is the standard test for real information | Information quality signal |
| 13 | Thompson (2022) | Block bootstrap | Bootstrap preserving temporal structure | scripts/16_bootstrap_validation.py | Independent bootstrap ignores autocorrelation | More accurate CIs |
| 14 | Vovk et al. (2005) | Conformal prediction | Distribution-free prediction intervals | analysis/conformal_prediction.py | Finite-sample coverage guarantees | Uncertainty-aware betting |
| 15 | Bailey & De Prado (2012) | Deflated Sharpe | Multiple testing correction | optimization/iterative_controller.py | Many tests inflate apparent success | Honest performance claims |

## Methods Evaluated and Rejected

| # | Method | Why Rejected | Evidence |
|---|--------|-------------|----------|
| 1 | Full Kelly staking | Ruin probability >10% with estimation error | Thorp (2006), simulation shows 23% path to zero |
| 2 | Class weighting for draws | Destroys overall accuracy | Empirical test: accuracy dropped from 54% to 38% |
| 3 | Synthetic-to-real transfer (deep nets) | Deep nets generalize poorly OOD | Transfer experiment: 42-47% vs 48-55% market |
| 4 | Random train/test split | Underestimates overfitting, leaks temporal info | Literature + empirical: 10-30% overoptimism |
| 5 | Accuracy-only model selection | Calibrated models with lower accuracy can have better ROI | Brier decomposition + betting simulation |
| 6 | Unregularized neural networks | Severe overfitting on <2000 matches | Train-test gap >15% with no regularization |

## Future Research Directions

| # | Area | Source | Potential Implementation |
|---|------|--------|------------------------|
| 1 | Beta calibration | Kull et al. (2017) | models/calibration.py — could capture inverted-S miscalibration |
| 2 | XGBoost with custom objective | 2025 thesis | Custom loss penalizing correlation with market odds |
| 3 | Graph-based team interaction | NeurIPS 2023 | Model team strength interactions as a graph |
| 4 | Mixture-of-experts | ICML 2024 | Regime-specific expert models for different league states |
| 5 | Online Bayesian updating | Murphy (2023) | Continuous model updating with new match data |
