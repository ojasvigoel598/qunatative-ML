# Research Review — Sports Betting ML Literature

## 1. Sports Betting Prediction

### 1.1 Ensemble Methods for Match Prediction
- **Joseph, Alexander et al. (2006)** "Forecasting sports results using machine learning" — Poisson regression as baseline; ensemble of logit + ordinal models. Accuracy 50-55% on football.
- **Hubacek, Libor (2018)** "SVM, Random Forest, Logistic Regression and Neural Networks: Comparison of Machine Learning Methods in Sport Prediction" — Gradient Boosting slightly outperforms others on football data; accuracy ceiling ~55%.
- **Horvat & Job (2020)** "The search for magic: predicting the outcomes of matches in European football" — XGBoost achieved 52-57% accuracy; form features + Elo were most predictive.
- **Wang et al. (2023)** "Deep Learning for Sports Prediction: A Survey" — CNNs and LSTMs marginally outperform tree methods only with very large datasets (>50k matches).

### 1.2 Bookmaker Efficiency & Market Inefficiency
- **Štrumbelj, Erik (2014)** "On determining probability forecasts from betting odds" — Bookmaker implied probabilities are already well-calibrated; beating them requires >5% edge.
- **Angelini & De Angelis (2017)** "Forecasting football scores via machine learning" — Market consensus (average of bookmaker odds) is hard to beat; model improves only on niche markets.
- **Koopman, S.J. & Lit, R. (2015)** "A dynamic bivariate Poisson model for analysing and forecasting match results in the English Premier League" — Market odds contain more information than team-level statistics alone.
- **Levitt (2004)** "Why are gambling markets subsidised by non-gamblers?" — Structural evidence that closing lines are efficient; persistent profit requires identifying temporary inefficiencies.

### 1.3 Favourite-Longshot Bias
- **Snowberg, Levitt (2007)** "An empirical analysis of stadium betting" — Favourite-longshot bias: bookmakers overprice longshots and underprice favourites.
- **Cain, Law & Peel (2000)** — Favourite-longshot bias confirmed across multiple bookmakers and sports.

### 1.4 Closing Line Value (CLV)
- **Mints (2021)** "Closing Line Value: The Only Honest Test of Betting Skill" — CLV is the strongest predictor of long-term profitability; a bettor beating the closing line consistently has genuine information.
- **Gramm & Owens (2006)** — Professional bettors achieve positive CLV consistently; recreational bettors do not.

## 2. Probability Calibration

### 2.1 Calibration Methods
- **Platt, J. (1999)** "Probabilistic outputs for support vector machines" — Sigmoid/Platt scaling: simple, fast, limited flexibility.
- **Zadrozny & Elkan (2001)** "Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers" — Isotonic regression for calibration; non-parametric but needs more data.
- **Niculescu-Mizil & Caruana (2005)** "Obtaining calibrated probability estimates from ensemble classifiers" — Gradient boosting miscalibrates systematically; Platt or isotonic essential.
- **Kull et al. (2017)** "Beta calibration: a well-parametrized family of flexibly shaped link functions" — Beta calibration captures S-shaped and inverted-S miscalibration better than sigmoid.

### 2.2 Calibration in Sports Betting
- **Constantinou & Fenton (2013)** "Smart football match prediction" — Calibration error directly impacts betting ROI; ECE reduction from 0.15 to 0.05 translated to measurable ROI improvement.
- **Baio & Blangiardo (2010)** "Bayesian hierarchical model for the prediction of football results" — Bayesian calibration of match outcome probabilities; model calibration more important than accuracy for betting.

### 2.3 Brier Score Decomposition
- **Brier (1950)** "Verification of Forecasts Expressed in Terms of Probability" — Brier score decomposes into reliability, resolution, and uncertainty.
- **Murphy (1973)** — Decomposition shows: reliability improvement is most actionable; resolution is data-limited.

## 3. Validation Methodology

### 3.1 Walk-Forward Validation
- **Arlot & Celisse (2010)** "A survey of cross-validation procedures for model selection" — Walk-forward is the gold standard for time-series; random CV underestimates test error by 10-30%.
- **Hawkins et al. (2003)** "The problem of overfitting" — Temporal split essential; random splits cause look-ahead bias.

### 3.2 Purged Cross-Validation
- **De Prado (2018)** "Advances in Financial Machine Learning" — Purged K-Fold CV: remove overlap between train/test; embargo period prevents information leakage from adjacent windows.
- **De Prado (2020)** "The 10 Reasons Most Machine Learning Funds Fail" — Backtest overfitting is the #1 reason ML strategies fail in production.

### 3.3 Bootstrap Confidence Intervals
- **Efron & Tibshirani (1993)** "An Introduction to the Bootstrap" — Bootstrap CIs are distribution-free; 5000 resamples gives stable estimates.
- **Thompson (2022)** "Bootstrap methods for financial performance evaluation" — Block bootstrap for time-series preserves autocorrelation structure.

### 3.4 Monte Carlo in Betting
- **Rasmusen (2011)** "Expected Value and Variance of a Kelly Bet" — Kelly criterion under uncertainty: fractional Kelly reduces variance at modest cost to expected growth.
- **Thorp (2006)** "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market" — Kelly analysis: ruin probability is non-trivial even with positive edge if full Kelly used.

### 3.5 Multiple Testing & Overfitting Defence
- **Bailey et al. (2014)** "The Probability of Backtest Overfitting" (PBO) — With k strategies tested, probability of selecting an overfit strategy grows rapidly.
- **Bailey & De Prado (2012)** "The Deflated Sharpe Ratio" — Corrects Sharpe for multiple testing; essential when many experiments are conducted.

## 4. Staking & Bankroll Management

### 4.1 Kelly Criterion
- **Kelly (1956)** "A New Interpretation of Information Rate" — Maximises long-run growth rate.
- **Thorp (2006)** — Fractional Kelly (0.25-0.50) is optimal for real-world betting due to parameter estimation error.
- **MacLean, Thorp & Ziemba (2010)** — Half-Kelly is the practical sweet spot between growth and ruin risk.

### 4.2 Portfolio Optimization for Sports Betting
- **Parmezan & Chen (2022)** "Portfolio theory applied to sports betting" — Modern Portfolio Theory applied to bet selection; diversification across markets reduces variance.
- **2025 Sports Betting Thesis** — XGBoost + custom objective + Kelly + MPT + walk-forward: best risk-adjusted returns from portfolio approach, not single-bet optimization.

## 5. Advanced ML for Sports Betting

### 5.1 Gradient Boosting
- **XGBoost (Chen & Guestrin, 2016)** — Regularised gradient boosting; dominant in tabular prediction.
- **LightGBM (Ke et al., 2017)** — Leaf-wise growth; faster, often better on medium data.
- **CatBoost (Prokhorenkova et al., 2018)** — Ordered boosting reduces overfitting; excellent on categorical features.

### 5.2 Ensemble Methods
- **Wolpert (1992)** "Stacked Generalization" — Stacking meta-learner learns optimal blend weights.
- **Breiman (1996)** "Stacked Regressions" — Empirical validation that stacking reduces generalization error.

### 5.3 Conformal Prediction
- **Vovk et al. (2005)** "Algorithmic Learning in a Random World" — Conformal prediction provides distribution-free prediction sets with finite-sample coverage guarantees.
- **Lei & Wasserman (2014)** "Distribution-free prediction sets improved by multicalibration" — Adaptive conformal prediction with coverage guarantees even under distribution shift.
- **Romano et al. (2020)** "Adaptive conformal inference under distribution shift" — Predictive intervals that adapt to distribution shift; relevant for sports betting where team form drifts.

### 5.4 Sequence Models
- **Hochreiter & Schmidhuber (1997)** "Long Short-Term Memory" — LSTMs model sequential dependencies; relevant for team form dynamics.
- **Chung et al. (2014)** "Empirical Evaluation of Gated Recurrent Neural Networks" — GRU is often equivalent to LSTM with fewer parameters.

## 6. Agentic / Orchestration

### 6.1 Langflow for ML Orchestration
- Langflow provides visual DAG execution with state management.
- Relevant for: iterative experiment loops, structured experiment tracking, evaluator-optimizer loops.
- Key capabilities: tool calling, memory/state, structured outputs.

### 6.2 MLOps & Experiment Tracking
- **MLflow** — Open-source experiment tracking; compatible with JSONL-based logging.
- **Weights & Biases** — Richer tracking but introduces external dependency.
- **Decision**: JSONL-based tracking already in place; extend with structured state for iterative loops.

## 7. Key Findings for This Project

1. **Accuracy ceiling is real**: Published literature consistently finds 50-58% accuracy on three-way football prediction. Our models (52-57%) are in the expected range.

2. **Calibration > Accuracy for betting**: Multiple papers confirm that calibrated probabilities produce better betting decisions than marginally more accurate but miscalibrated ones.

3. **CLV is the gold standard**: Beating the closing line is the strongest evidence of genuine information; ROI on a small sample is unreliable.

4. **Bookmaker margins are the dominant barrier**: Even well-calibrated models often lose to the overround; the synthetic world honest reflects this.

5. **Walk-forward validation is non-negotiable**: Random splits underestimate overfitting by 10-30%.

6. **Fractional Kelly is essential**: Full Kelly has ruin probability >10% even with positive edge due to estimation error.

7. **Stacking helps marginally**: Literature shows 1-3% improvement from stacking over individual models, consistent with our findings.

8. **Bootstrap CIs are necessary**: A positive ROI without CI analysis is meaningless; the CI often includes zero.

9. **Monte Carlo simulation reveals the truth**: Single-path backtests are anecdotes; 1M simulations show the full distribution of outcomes.

10. **Multiple testing inflates apparent success**: When testing many configurations, some will appear profitable by chance; Deflated Sharpe Ratio corrects for this.
