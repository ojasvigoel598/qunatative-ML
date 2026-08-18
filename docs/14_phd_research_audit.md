# Quantitative Sports Betting AI — Research Audit, Benchmarking & Code Adaptation

**Audit date:** 2026-08-18  
**Repository:** `sports_betting_model`  
**Scope:** source-code audit, primary-source literature review, serious open-source review, leakage/backtest red-team, and minimal high-confidence code corrections.

> **Important conclusion:** this repository is a credible research demonstrator, not evidence of a live positive-EV betting operation. Its strongest contribution is validation discipline and honest negative findings. The current evidence does not establish a robust out-of-sample financial edge after margin, execution, and market comparison.

## Executive Summary

The system is an unusually complete small research platform: it has a synthetic world, Poisson/Elo pricing, a calibrated tree layer, uncertainty propagation, odds and closing-line fields, chronological real-data experiments, a walk-forward agent, staking stress tests, CLV diagnostics, and 37 automated tests. That is materially stronger than the typical sports-prediction notebook.

The economic evidence is weaker than the model-quality evidence. The canonical synthetic run reports 23–26 selected bets, negative ROI, and CLV close to zero. The real-data experiments show useful predictive structure—roughly 50–55% three-way accuracy in the reported seasons—but the market remains the strongest probability benchmark and the real-data betting samples are too small for a stable profitability conclusion. Estimated edge is not realized edge; a model can be better than a naive baseline and still fail to overcome overround.

The highest-value corrections are methodological rather than architectural:

1. **One point-in-time Elo state:** `pipeline.train_models()` previously prepared Elo features and then called `poisson.train()`, which prepared and advanced Elo a second time. Training features, fitted regressions, and inference state could therefore disagree. Training now owns one sequential pass and exposes `training_features`.
2. **Temporal ML diagnostics/calibration:** the ML layer previously used a random holdout and default random-style calibration folds. It now uses a chronological 80/20 diagnostic holdout and `TimeSeriesSplit` for sigmoid calibration.

These changes do not create edge; they reduce a source of experimental ambiguity. The next major work should be a pre-registered, market-aware, nested walk-forward evaluation with block-bootstrap uncertainty and a sacred holdout period—not a larger neural network.

## Current-System Assessment

### Architecture and data flow

The core flow is:

```text
synthetic generator or football-data.co.uk CSV
        ↓
date sort / Elo / shifted rolling form
        ↓
65% chronological train · 15% validation · 20% test
        ↓
Poisson goals + Elo  ─┐
                     ├─ average probability ensemble
calibrated GB        ┘
        ↓
edge = probability × decimal odds − 1
        ↓
thresholds: edge, odds floor, probability floor
        ↓
quarter-Kelly or Q-learning Kelly multiplier
        ↓
settlement, equity curve, ROI, drawdown, CLV, probability metrics
```

The repository also contains separate experimental paths:

- `models/adaptive_model.py`: online Elo/form state and scheduled/drift-triggered ML refits.
- `models/dynamic_thinking.py`: model/market fusion, multi-book dispersion, confidence and drawdown rules.
- `models/lstm_model.py`, `models/nn_model.py`, `models/tf_hybrid.py`: optional neural/state-space experiments.
- `agent_sim/`: chronological multi-league reveal, leakage audit, per-run ledgers, frozen validation window.
- `scripts/14_tennis_walkforward.py`: a different-sport two-outcome walk.

### Targets

- Football: three-class 1X2 result (`H`, `D`, `A`) and auxiliary home/away goals.
- Tennis: winner/loser two-outcome target.
- Synthetic football: goal counts are generated from independent Poisson processes and 1X2 is derived from the score.

### Data

1. **Synthetic data** in `pipeline.py`: ten latent-strength teams; goals from a Poisson world; opening/closing odds from the world probabilities plus margin, noise and favourite–longshot shading; a second synthetic book supports price shopping.
2. **Football data** in `data/real_data.py`: football-data.co.uk results, B365/Pinnacle-style odds, closing and maximum-price columns, plus optional shots/cards/corners.
3. **ATP data** in `agent_sim/tennis.py`: tennis-data.co.uk yearly workbooks, with B365/Pinnacle and maximum odds.

The raw real-data path is cached locally. Data provenance is documented, but there is no dataset hash/manifest, immutable snapshot ID, schema version, or feature-store version. That is a reproducibility gap for future work.

### Feature engineering

- Sequential Elo ratings, updated only after observed results.
- Poisson regression features: home and away Elo.
- ML features: shifted rolling home/away goals averages.
- Adaptive features: Elo difference, rolling goals and recent points.
- Sequence features: historical goals/results and optional shots, corners, cards and odds.
- Market features in the dynamic/hidden-signal paths: devigged public/sharp probabilities, consensus, dispersion and split.

Positive design choices are the explicit `.shift(1)` rolling-feature rule, online state updates and stored leakage timestamps. The core model does not use injuries, lineups, xG, weather, referee, travel, liquidity, limits, or execution latency.

### Training and validation

- Core split: chronological 65/15/20.
- Real season script: expanding-window season evaluation and online feature construction.
- Agent simulation: chronological reveal and an optional frozen final window.
- Deep models: mostly chronological training/validation walks, with early stopping in the sequence model.
- Hyperparameters are mostly hand selected. There is no nested temporal hyperparameter search, no pre-registered candidate universe, and no multiple-testing correction over the many experiments.

The ML layer’s previous random diagnostic split has been replaced by a chronological holdout and temporal calibration folds. This is a diagnostic improvement; the outer test remains the key evidence.

### Probability estimation and calibration

- Poisson score-grid probabilities, optionally Dixon–Coles corrected for low scores.
- Gradient Boosting wrapped in sigmoid calibration.
- ECE, multiclass Brier, log loss, reliability curves and model-vs-market comparisons.
- Monte-Carlo coefficient sampling for Poisson parameter uncertainty.
- `models/calibration.py` also includes isotonic helpers, but its `isotonic_fit` implementation is not a replacement for proper out-of-fold or rolling calibration: fitting each class primarily on non-winning rows is not a defensible production calibration protocol.

The code correctly treats probability quality as more important than raw accuracy for EV calculation, but calibration is not yet evaluated specifically in the selected betting region with confidence intervals and enough independent folds.

### Odds, margin and EV

- Decimal odds are converted through `1 / odds` and normalized to remove overround for market probability comparisons.
- Bet selection uses `p × odds − 1` against opening/public prices.
- Price shopping is supported.
- Closing odds are recorded for CLV.

Caveats:

- Normalizing inverse odds is a practical devigging rule, not the uniquely correct bookmaker-probability recovery method. Alternative power or Shin-style devigging should be a sensitivity analysis.
- The core backtest does not model odds availability, bet limits, rejected bets, timing, line latency, account restrictions, commission, currency, or slippage.
- Best opening price and best closing price are not automatically guaranteed to be available at the same book or at the same timestamp. A production ledger needs bookmaker identity and timestamp per quoted price.

### Selection, staking and portfolio risk

- Current selection: choose the maximum edge, then require edge threshold, minimum odds and minimum model probability.
- Current sizing: quarter-Kelly with caps, or Q-learning over Kelly multipliers.
- Stress tests compare flat and Kelly policies.
- No portfolio covariance, mutual-exposure limit, market/league concentration limit, or simultaneous-bet allocation exists in the core pipeline. The agent has daily caps, but that is not a correlation model.
- RL is trained on a small validation discovery path and repeats the same experiences across episodes. It is therefore an experiment in policy parametrization, not evidence that RL found a generalizable optimal policy.

### Backtest metrics and monitoring

The system reports ROI, yield-like profit measures, profit factor, strike rate, average edge, CLV, CLV t-statistic, Sharpe, CAGR and maximum drawdown, plus log loss, Brier, accuracy and ECE. The data-size sweep and loss-attribution reports are useful diagnostics.

Missing production controls include:

- block-bootstrap confidence intervals by season/week/league;
- a sacred untouched holdout opened only once;
- a complete experiment registry and configuration hash;
- model/data version IDs in every prediction row;
- live-vs-backtest drift and calibration monitoring;
- alerting for odds freshness, missing data and schema changes;
- explicit order/execution simulation;
- portfolio-level exposure and correlation accounting.

## Quantitative Betting Research

### What serious systems optimize

A betting model is a decision system, not merely a classifier. The hierarchy should be:

1. **Proper probability quality:** log loss and Brier, with reliability by probability bucket and by selected-bet region.
2. **Incremental market information:** model minus devigged market log loss/Brier and a price-timestamped CLV benchmark.
3. **Economic value:** realized profit after margin, realistic prices, limits and slippage.
4. **Risk-adjusted growth:** fractional Kelly or constrained portfolio growth with drawdown and ruin controls.
5. **Robustness:** performance across seasons, leagues, bookmakers, price snapshots and pre-specified subgroups.

Accuracy answers “which class is most likely?” It does not answer “is the quoted price profitable?” A 55% classifier can lose at short prices if probabilities are miscalibrated, while a lower-accuracy model can make money if it identifies a small number of correctly priced long odds. This is why the repository’s own market comparison and CLV diagnostics are more informative than its headline accuracy.

### Market information: ignore, feature, prior or benchmark?

The defensible answer is **all four, but in separate experiments**:

- **Benchmark:** always compare to devigged opening and closing market probabilities.
- **Feature/prior:** use only information available at the decision timestamp; compare a model-only forecast to a market-aware forecast.
- **Mispricing detector:** bet only when the model’s incremental signal survives calibration, uncertainty, price sensitivity and a market benchmark.
- **Closing line:** use as a post-decision information benchmark, never as a feature for the same pre-match decision.

The current dynamic layer uses the market as a signal, while the canonical core mostly uses it only after prediction. That separation is valuable, but it needs a single reproducible experiment matrix rather than many independently tuned scripts.

### Ensembles

Simple averaging is a credible starting point because Poisson/Elo and tree models have different inductive biases. However, fixed 50/50 averaging is not evidence-based unless validated out of sample. The next defensible comparison is:

- Poisson/Elo only;
- calibrated ML only;
- market only;
- fixed blend weights selected on inner temporal folds;
- constrained stacking trained only on inner folds;
- market-aware blend with weights allowed to vary by league/season only when the outer protocol supports it.

Dynamic weights should be judged against a frozen-weight control and a market-only control. A more complex meta-learner is not automatically superior.

### Uncertainty

The Poisson parameter Monte Carlo measures one uncertainty source: coefficient-estimation uncertainty conditional on the model. It does not fully capture:

- model-form uncertainty;
- team-strength uncertainty;
- non-stationarity/concept drift;
- data revision and odds uncertainty;
- aleatoric match randomness.

For betting, uncertainty should affect the decision through a conservative lower confidence bound on edge, probability shrinkage toward a market/base-rate prior, and stake reduction—not by turning a standard deviation into a claim of calibrated coverage. A rolling conformal or block-bootstrap study could be useful, but ordinary iid conformal guarantees do not automatically hold under sports time dependence.

### Walk-forward and backtest bias

Chronological validation is mandatory. Expanding windows are appropriate when old data remains useful; rolling windows are appropriate when drift makes old data harmful. Nested temporal validation is required whenever thresholds, features, model families, ensemble weights or stake rules are selected from historical data.

The major biases to audit are:

- look-ahead from post-match form, final rankings, closing odds or future availability;
- preprocessing fit on future rows;
- duplicate fixtures and re-ordered matches;
- using the same season to select and evaluate thresholds;
- repeated experiments on the same test period;
- best-price selection without timestamp/book identity;
- survival bias from only retaining available markets;
- naive iid confidence intervals when bets are clustered by team, day, league or season.

## Academic Literature Review

| Technique / question | Verified source | What it supports | Transfer to this repository |
|---|---|---|---|
| Poisson score model and low-score dependence | M. J. Dixon & S. G. Coles (1997), *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, JRSS-C 46(2), 265–280. DOI: [10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) | Parametric score modelling and a low-score correction are credible football baselines; betting profitability must still be evaluated against market prices. | Already adapted in `models/poisson_elo_model.py`; use as a probability baseline, not a profitability claim. |
| Economic value of statistical forecasts | M. J. Dixon & P. Pope (2004), *The value of statistical forecasts in the UK association football betting market*, IJF 20(4), 697–711. DOI: [10.1016/j.ijforecast.2003.12.007](https://doi.org/10.1016/j.ijforecast.2003.12.007) | Forecasts should be compared directly with bookmaker prices, including bookmaker differences and potential arbitrage. | Supports book-specific prices, market benchmark, CLV and price-shopping experiments. |
| Elo as an informative covariate | L. M. Hvattum & H. Arntzen (2010), *Using ELO ratings for match result prediction in association football*, IJF 26(3), 460–470. [Verified bibliographic record](https://ideas.repec.org/a/eee/intfor/v26yi3p460-470.html) | Elo-derived covariates are a serious benchmark and should be compared economically and statistically with alternatives. | Already adapted; the new single-pass fix makes its temporal state coherent. |
| Proper scoring rules | T. Gneiting & A. E. Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102, 359–378. DOI: [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437) | Log loss and Brier reward honest probabilistic forecasts; accuracy alone is insufficient. | Supports the existing metrics and the recommendation to optimize probability quality before EV. |
| Calibration vs accuracy for betting | C. Walsh & A. Joshi (2023/2024 version), *Machine learning for sports betting: should model selection be based on accuracy or calibration?*, arXiv:[2303.06021](https://arxiv.org/abs/2303.06021) | In their NBA experiment, calibration-based selection outperformed accuracy-based selection economically; the result is sport/data/protocol-specific, not universal. | Supports temporal calibration diagnostics and calibration-by-betting-region; do not copy the reported ROI as a general law. |
| Kelly and growth-optimal sizing | J. L. Kelly Jr. (1956), *A New Interpretation of Information Rate*. DOI: [10.1002/j.1538-7305.1956.tb03809.x](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x) | Kelly maximizes expected log-growth under strong assumptions about known probabilities and repeated opportunities. | Supports fractional/capped Kelly only after probability uncertainty and dependence are addressed. |
| Practical sports-betting risk controls | M. Uhrín, G. Šourek, O. Hubáček & F. Železný (2021), *Optimal sports betting strategies in practice: an experimental review*, arXiv:[2107.08827](https://arxiv.org/abs/2107.08827), related DOI [10.1093/imaman/dpaa029](https://doi.org/10.1093/imaman/dpaa029) | Risk-control modifications matter; their unified tests found adaptive fractional Kelly practical across settings. | Supports capped fractional-Kelly experiments; does not justify the current Q-learning agent. |
| Betting decision theory | J. P. Dmochowski (2023), *A statistical theory of optimal decision-making in sports betting*, PLOS ONE 18(6), e0287601. DOI: [10.1371/journal.pone.0287601](https://doi.org/10.1371/journal.pone.0287601) | Selecting which events to bet requires more than a point prediction; positive expectation depends on relevant outcome quantiles and bookmaker proposition. | Supports modeling the full distribution/uncertainty and explicitly modeling the no-bet region. |
| Dynamic state-space ratings | M. E. Glickman & H. S. Stern (1998), *A State-Space Model for National Football League Scores*, JASA 93, 25–35. [Author-hosted PDF](https://glicko.net/research/nfl.pdf) | Latent team strength can evolve over time; state-space structure is a principled alternative to ad hoc refits. | A Tier-2 experiment after the baseline is fixed; do not deploy LSTM first. |
| Data snooping | H. White (2000), *A Reality Check for Data Snooping*, Econometrica. DOI: [10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152) | Reusing a dataset for model/strategy selection inflates apparent significance; bootstrap reality checks address a family of candidates. | Supports a candidate registry, nested selection and multiple-testing correction. |
| Backtest overfitting | D. H. Bailey et al. (2015), *The Probability of Backtest Overfitting*, SSRN:[2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253); D. H. Bailey & M. López de Prado (2014), *The Deflated Sharpe Ratio*, SSRN:[2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) | The best historical strategy among many candidates can be a selection artifact; performance statistics need selection-aware interpretation. | Supports sacred holdout, block bootstrap and reporting how many variants were tried. |

No paper above is evidence that a generic neural network, transformer or reinforcement learner will beat a mature bookmaker market. The literature supports disciplined probability and decision evaluation more strongly than complexity.

## GitHub / Open-Source Review

The repositories below were inspected at the README/license level through their public GitHub pages. No code was copied.

| Repository | License / evidence | Actual implementation observed | Reusable lesson | What was adapted |
|---|---|---|---|---|
| [georgedouzas/sports-betting](https://github.com/georgedouzas/sports-betting) | MIT, copyright notice in repository `LICENSE` | DataLoader separates statistics and odds sources; scikit-learn estimator wrapper; `ClassifierBettor`; `backtest` accepts `TimeSeriesSplit`; CLI and API expose model, odds, stake and CV. | Separate data provenance, model fitting, odds, bettor and evaluation; use temporal CV as a first-class parameter. | Conceptual comparison only; no code reused. |
| [Volodymyr4K/market-efficiency-lab](https://github.com/Volodymyr4K/market-efficiency-lab) | MIT code license; README explicitly excludes third-party data redistribution | Walk-forward OOS studies across MLB/tennis; closing line as benchmark; block-bootstrap CIs by season; minimum detectable effect; sacred holdout; provisional labeling; negative results retained. | The validation machinery is more valuable than the model brand. Clustered/block uncertainty and a holdout prevent false precision. | The report’s proposed experiment design is directly informed by these public methodological practices; no source code reused. |
| [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) | No `LICENSE` file was verified at the linked path; do not reuse code without permission. | SQLite data/odds ingestion, matchup features, XGBoost/NN training scripts, EV and optional Kelly, Flask UI. | A deployable pipeline must preserve odds provenance and separate ingestion, training and serving. | Not adapted; claims and code are not used as evidence. |

The project’s existing `docs/06_research_notes.md` names additional repositories, but their implementations and licenses were not sufficiently verified in this audit, so they are not treated as primary evidence.

## Technology Comparison Matrix

| Research technique | Evidence | Used by this code? | Implementation quality | Potential benefit | Difficulty | Priority |
|---|---|---:|---|---|---:|---:|
| Poisson / Dixon–Coles score model | Strong football literature | Yes | Good baseline; DC rho is fitted but full likelihood treatment is limited | Better low-score probabilities | Low | High |
| Sequential Elo / dynamic rating | Strong benchmark literature | Yes | Good, now single-pass in core | Stable strength signal | Low | High |
| Shifted rolling features | Leakage principle | Yes | Good unit test; broader feature provenance still needed | Prevent artificial lift | Low | High |
| Chronological train/test | Essential | Yes | Good core/real scripts | Honest OOS estimate | Low | High |
| Temporal calibration folds | Calibration + time-series principle | **Yes, newly corrected** | Good first step; needs outer rolling calibration evaluation | More defensible EV | Low | High |
| Market-only benchmark | Strong empirical practice | Yes | Present, but not universal across every experiment | Detect incremental signal | Low | High |
| Market as a prior/feature | Strong practical rationale; not guaranteed edge | Dynamic layer only | Experimental, not one source of truth | Better pricing and smaller model variance | Medium | High experiment |
| CLV | Strong practitioner/market diagnostic; must be timestamped | Yes | Useful, but synthetic and small real samples limit inference | Earlier edge validation | Low | High |
| Calibration by selected region | Decision theory | Partial | Loss attribution has a calibration gap; no robust confidence interval | Reduce winner’s curse | Medium | High |
| Parameter uncertainty | Statistical rationale | Yes | Coefficient MC only; not full predictive uncertainty | Filter/stake reduction | Medium | Medium |
| Block bootstrap by season | Backtest research practice | No | Missing | Honest ROI/CLV intervals under clustering | Medium | High |
| Nested temporal CV / sacred holdout | Data-snooping literature | Partial | Expanding tests exist; thresholds and variants are not centrally pre-registered | Reduce selection bias | Medium | High |
| Fractional Kelly | Kelly/practical betting literature | Yes | Capped and stress-tested; probability error/dependence not fully modeled | Control drawdown | Low | High |
| Correlated portfolio allocation | Finance / Kelly extension | No | Missing | Limit same-team/league exposure | Medium | High |
| Online drift detection | Adaptive forecasting | Yes | Experimental; refit trigger itself needs a pre-specified audit | Handle non-stationarity | Medium | Medium |
| Bayesian state-space rating | Dynamic-rating literature | No | LSTM is not an equivalent Bayesian SSM | Better latent-strength uncertainty | High | Medium |
| Stacking / dynamic ensemble weights | Ensemble literature | Partial | Fixed average in core, adaptive blend in experimental layer | Incremental forecast value | Medium | Medium |
| Conformal uncertainty | Modern uncertainty literature | No | iid guarantees do not directly transfer to dependent matches | Conservative decision sets | High | Low/experimental |
| Reinforcement learning for selection/sizing | Limited credible evidence relative to supervised baselines | Yes, Q-learning demo | Small repeated validation sample; high overfit risk | Policy research only | Medium | Avoid as production default |
| Transformers / large deep nets | No repository-specific evidence of superiority | Experimental models exist | Data volume is too small and market already strong | Possible representation learning at scale | High | Avoid now |

## Mathematical Techniques and Applicability

### Probability and edge

For decimal odds `o` and model probability `p`, the unit-stake expected profit is:

```text
EV = p(o - 1) - (1 - p) = p o - 1
```

This is correct only for the exact price actually available. If `p` is biased upward by selection or calibration error, the apparent EV is biased upward. The decision should therefore include a market prior, uncertainty shrinkage and a minimum detectable effect.

### Kelly and fractional Kelly

For a binary wager with net odds `b = o - 1`, win probability `p` and loss probability `q`, full Kelly is:

```text
f* = (b p - q) / b = EV / b
```

Full Kelly assumes `p` is known, outcomes are modeled correctly, capital can be reallocated as assumed, and opportunities/dependence are handled. The repository’s quarter-Kelly and caps are more defensible than full Kelly. A stronger future implementation should size a simultaneous portfolio under a total exposure constraint and use a lower confidence bound for `p`.

### Calibration

For each outcome bucket, compare mean predicted probability with observed frequency. Use rolling or season-block reliability, log loss, Brier and selected-region calibration. ECE is useful descriptively but is bin-sensitive and should not be the sole calibration criterion.

### Uncertainty

The current coefficient-Monte-Carlo method samples regression parameters and propagates them through the score grid. This is useful but conditional. A practical next layer is an ensemble/bootstrap distribution of complete forecasts, with a temporal calibration window and a conservative edge rule such as:

```text
lower_edge = mean_edge - z × edge_standard_error
bet only when lower_edge > minimum_detectable_edge
```

The `z` value and minimum effect must be selected inside training folds, not after inspecting the final test.

### Multiple testing and statistical significance

A t-statistic on 23 selected bets does not prove a strategy. Bets are selected, clustered and potentially dependent; many thresholds, policies and scripts were tried. Report:

- number of candidate variants;
- outer test sample and number of bets;
- block-bootstrap ROI/CLV distribution;
- probability of positive ROI under the pre-specified null;
- season-by-season signs;
- market-only and no-bet baselines.

## Model Architecture Comparison

| Layer | Current | Defensible next architecture |
|---|---|---|
| Structural forecast | Elo + two Poisson regressions + score grid | Keep as the anchor; test DC and dynamic state-space variant head-to-head. |
| Flexible forecast | Calibrated Gradient Boosting on Elo/form | Keep as a challenger; fit/calibrate temporally and compare to market. |
| Market | Mostly benchmark; dynamic layer fuses market | Make a market-only and market-aware blend first-class, using opening snapshot only. |
| Ensemble | Fixed average in core; adaptive blend experimentally | Select blend weights only in inner temporal folds; keep market-only control. |
| Uncertainty | Poisson coefficient Monte Carlo | Add block bootstrap/ensemble uncertainty and evaluate coverage/calibration. |
| Decision | Max edge + floors | Require robust lower-bound edge, minimum effect and timestamped price. |
| Sizing | Quarter Kelly / Q-learning | Keep fractional Kelly; replace Q-learning default with constrained, pre-registered policy comparison. |
| Portfolio | Per-bet independent | Add same-team, same-day, league and outcome exposure caps; model correlation. |
| Monitoring | CSV reports | Versioned data/model/config ledger and forward paper-trading record. |

## Market and Probability Modeling Recommendation

Do not ignore bookmaker odds. Use them in a controlled hierarchy:

1. **Opening market probability** is the baseline and a prior candidate.
2. **Model-only probability** measures whether football features add information.
3. **Market-aware probability** combines model and opening market through a blend or log-odds prior selected on inner temporal folds.
4. **Closing odds** are excluded from decision features and retained for CLV evaluation.
5. **Price shopping** records bookmaker and timestamp, not just the maximum number.
6. **Stake selection** uses only prices that were actually observable at the decision timestamp.

The current synthetic world should include an experiment where the model is trained without odds, then compared against market-only and model+market variants on the same forward paths. The result should be reported even if the market-aware system wins by simply tracking the market; that is not a modeling failure, but it is not an independent edge.

## Validation and Backtesting Protocol

### Baseline and models

- **Baseline:** current corrected Poisson/Elo + calibrated GB ensemble, current selection and quarter-Kelly cap.
- **Model A:** market-aware probability blend using opening devigged probabilities, with blend weight selected only inside temporal validation.
- **Model B:** robust edge filter using temporal calibration plus uncertainty/bootstrap lower edge.
- **Model C:** combined market-aware, calibrated, uncertainty-filtered system with constrained fractional Kelly.

### Fairness rules

All models must use the same:

- raw match rows and data cutoff;
- bookmaker/price snapshot;
- closing reference line;
- seasons and leagues;
- minimum odds and stake cap search space;
- bankroll and settlement assumptions;
- outer test windows.

The outer holdout year is opened once. Every threshold, ensemble weight, calibration method and staking cap is chosen in an inner temporal loop. No experiment is allowed to rewrite the final report after seeing the outer result.

### Required metrics

For every fold and pooled result:

- log loss, multiclass Brier, accuracy, balanced accuracy, ECE and reliability curves;
- market-minus-model log-loss/Brier delta;
- selected-bet count, turnover, hit rate, average edge and betting-region calibration gap;
- yield/ROI on turnover, profit, profit factor;
- average and distribution of CLV, CLV win rate and block-bootstrap CLV interval;
- max drawdown, volatility, Sharpe/Sortino with documented annualization;
- risk of ruin and probability of loss;
- season/league/bookmaker breakdown;
- confidence intervals based on blocks, not iid bet rows.

## Ablation Study

Run in one frozen outer protocol:

```text
0. Current corrected baseline
1. Baseline + opening market benchmark only
2. Baseline + market-aware blend
3. Baseline + temporal calibration only
4. Baseline + uncertainty filter only
5. Baseline + richer pre-match features
6. Baseline + constrained ensemble/stacking
7. Baseline + fractional-Kelly policy alternatives
8. Combined system
```

For each row, publish the full metrics above and the number of candidate variants explored. The combined system is not accepted unless it improves out-of-sample probability quality and survives the same market/CLV and block-bootstrap tests.

## Red-Team Review

A hostile reviewer would ask:

- Is the synthetic bookmaker structurally related to the model’s data-generating process? **Yes.** Synthetic results are internal validation, not market evidence.
- Does a positive selected edge prove profit? **No.** Edge is model-implied and selected on disagreement.
- Is the sample sufficient? **No** for 23–26 canonical bets or five real seasons to claim stable ROI.
- Is CLV conclusive? **No** when synthetic closing lines are independently generated and real price timestamps/book identity are incomplete; it is a useful diagnostic, not a theorem.
- Is the model incremental beyond the market? **Not established.** The repository’s own market comparison often shows the market with lower loss.
- Is the RL learner independently validated? **Not yet.** It learns from a small repeated validation experience set.
- Is there multiple testing? **Yes.** Many scripts, thresholds, models and dynamic variants are present; a sacred holdout and candidate count are required.
- Does performance survive another league/sport? **Partially for predictive metrics, not proven for ROI.** Transfer results are mixed and sequence models do not clearly dominate.
- Are odds realistic? **Better than random odds, but execution, limits, price availability and latency are simplified.**
- Could the model simply learn bookmaker odds? **In the core, odds are not model features; in the dynamic/LSTM experiments, market use must be benchmarked explicitly.**

## Current System: Exactly Three Strengths

### 1. Temporal and leakage discipline

**Evidence:** chronological core split; shifted rolling features; sequential Elo; `agent_sim` feature/result timestamps and invalidation; two-phase same-day tennis walk; tests for rolling leakage and reproducibility.  
**Why strong:** this is the minimum condition for a meaningful sports forecast and is absent from many toy repositories.  
**Preserve:** keep prediction, reveal, update as separate operations; require a cutoff timestamp in every row; add dataset hashes and timestamped odds.

### 2. Separation of probability quality from betting economics

**Evidence:** `evaluate_probability_quality`, model-vs-market metrics, ECE/Brier/log loss, CLV, loss attribution and explicit negative ROI in README/docs.  
**Why strong:** it prevents a lucky bet log or accuracy headline from being mistaken for edge.  
**Preserve:** make market-only, no-bet and closing-line baselines mandatory in every experiment.

### 3. Reproducible, testable end-to-end engineering

**Evidence:** modular models, CLI scripts, cached data paths, seeded synthetic generator, output artifacts and 37 passing tests in the repository’s verification suite.  
**Why strong:** the project can be challenged, rerun and falsified rather than existing only as a notebook claim.  
**Preserve:** small commits, deterministic configs, test fixtures, immutable result manifests and explicit optional deep-learning dependencies.

## Current System: Exactly Three Weaknesses

### 1. No statistically credible proof of incremental economic edge

**Evidence:** canonical selected-bet ROI is negative on a tiny sample; CLV is near zero in the canonical synthetic report; real-data ROI and CLV results are spread over few seasons and market comparisons often favor the bookmaker.  
**Why it matters:** predictive lift over a majority class does not overcome overround.  
**Severity:** critical for any profitability claim.  
**Fix:** market-aware outer walk, timestamped prices, block-bootstrap ROI/CLV, sacred holdout and forward paper-trading ledger.  
**Validation:** require positive market-minus-model probability delta and positive block-bootstrap CLV before considering a live trial.

### 2. Selection, calibration and uncertainty are not yet decision-grade

**Evidence:** max-edge selection targets the region where estimation error is likely largest; the Q-learning agent repeats a small validation experience set; coefficient MC omits model/drift uncertainty; isotonic helper is not proper OOF temporal calibration.  
**Why it matters:** these mechanisms can make advertised EV much larger than realized EV.  
**Severity:** high.  
**Fix:** rolling calibration by outcome and selected region, robust lower-edge filtering, market prior/shrinkage, block bootstrap, and a constrained fractional-Kelly default.  
**Validation:** selected-region reliability curves, calibration confidence intervals, CLV and ROI against identical baselines.

### 3. Research reproducibility is not yet experiment reproducibility

**Evidence:** many scripts write mutable reports/results; no central experiment registry, dataset checksum, parameter manifest, candidate count or immutable model artifact; real data can change when re-fetched.  
**Why it matters:** repeated tuning can silently become backtest selection.  
**Severity:** high.  
**Fix:** version data snapshots, hash configs/features/models, record all candidates and reserve a sacred final period.  
**Validation:** rerun from the manifest and verify byte-identical predictions/metrics.

## Exactly Three Highest-Value Opportunities

### #1 — Market-aware probability and CLV-first evaluation

The market is already the strongest available baseline in the repository’s real experiments. Add opening devigged market probabilities as a controlled prior/feature experiment, retain model-only and market-only controls, and make price timestamp/book identity explicit. This is more likely to improve risk-adjusted ROI than another learner because it attacks the system’s demonstrated bottleneck: model probability quality relative to the market.

### #2 — Nested temporal selection with block-bootstrap inference

Build one evaluation engine that selects thresholds, calibration, blend weights and stake policy only in inner temporal folds, then scores a sacred outer period. Bootstrap whole seasons/weeks or other meaningful clusters. This directly addresses the largest false-positive risk and converts “positive ROI” from a point estimate into a robustness distribution.

### #3 — Constrained uncertainty-aware portfolio sizing

Keep fractional Kelly but replace per-bet independent sizing with lower-confidence edge, total exposure caps, same-team/league correlation controls and realistic simultaneous settlement. This is the most defensible route to lower drawdown without pretending the predictor has more information than it does.

## Exactly Three Things Not to Do Yet

### 1. Do not add a transformer or larger neural network

The available football sample is small relative to the parameter search, and the repository’s own LSTM/GRU/deep-transfer results do not clearly beat simpler baselines or the market. Add capacity only after richer, timestamp-valid data and a pre-registered ablation show incremental market-relative probability value.

### 2. Do not deploy reinforcement learning as the default staking policy

The current Q-table is trained on a small repeated validation sequence and has no independent policy holdout. Use it as a research challenger against fixed fractional Kelly and flat staking, not as evidence of learned optimal control.

### 3. Do not expand features or optimize historical ROI without a sacred holdout

Injuries, xG, weather, line movement and bookmaker signals may help, but each creates timestamp, revision and multiple-testing risks. Add one feature family at a time, log its availability cutoff, and test it only through nested temporal evaluation.

## Research-to-Code Adaptation

### Adaptation A — coherent sequential training state

- **Technique:** point-in-time dynamic rating with no repeated state mutation.
- **Source:** Elo/football forecasting practice; Hvattum & Arntzen (2010); leakage principles in the reviewed walk-forward systems.
- **Mathematical idea:** the feature for match `t` is a function of information `I_(t-1)`. A second Elo pass changes `I_(t-1)` and makes the fitted design matrix inconsistent with inference.
- **Previous implementation:** `train_models()` called `prepare_features()` and then `train()`, while `train()` called `prepare_features()` again.
- **New implementation:** `PoissonEloModel.train()` resets state, performs one pass, stores `training_features`; `pipeline.train_models()` consumes that stored frame.
- **Files:** `models/poisson_elo_model.py`, `pipeline.py`, `tests/test_pipeline.py`.
- **Expected benefit:** valid feature/state alignment and reproducible retraining; not an expected direct ROI increase.
- **Risk:** callers relying on a previously advanced model state during retraining could change behavior; fresh training is the intended contract.
- **Validation:** regression test compares features and final Elo ratings against an independent single-pass reference.

### Adaptation B — temporal ML calibration and diagnostics

- **Technique:** chronological holdout and time-series calibration folds.
- **Source:** Gneiting & Raftery (2007) for proper probabilistic evaluation; Walsh & Joshi (arXiv:2303.06021) for calibration’s economic relevance; temporal validation practice in georgedouzas/sports-betting and market-efficiency-lab.
- **Mathematical idea:** calibration maps scores to probabilities using only information available before the evaluation period. Randomly mixing dates can make a diagnostic describe a different information regime.
- **Previous implementation:** random `train_test_split` and default calibration CV.
- **New implementation:** `TimeSeriesSplit(n_splits=3)` for sigmoid calibration and chronological 80/20 diagnostic holdout.
- **Files:** `models/ml_layer.py`, `tests/test_pipeline.py`.
- **Expected benefit:** better-aligned probability diagnostics and less optimistic temporal interpretation.
- **Risk:** temporal folds have less effective sample size and can be noisier; this is a feature of honest uncertainty, not a defect.
- **Validation:** full outer walk, reliability curves, log loss/Brier, market comparison and selected-region calibration.

### Proposed Adaptation C — market prior (not yet implemented)

- **Technique:** opening-market benchmark plus constrained model/market blend.
- **Source:** Dixon & Pope (2004), Walsh & Joshi (2023/24), market-efficiency-lab methodology.
- **Mathematical idea:** combine independent model log odds with devigged market log odds, with blend weight selected in inner folds; closing odds remain evaluation-only.
- **Current implementation:** market is benchmarked in the core and fused in `DynamicThinkingLayer`, but not one canonical controlled experiment.
- **Validation:** model-only vs market-only vs blend on identical folds and prices; accept only if incremental log loss/Brier and block-bootstrap CLV improve.

### Proposed Adaptation D — block-bootstrap and sacred holdout (not yet implemented)

- **Technique:** cluster-aware uncertainty and selection-bias control.
- **Source:** White (2000); Bailey et al. (2014/2015); market-efficiency-lab README methodology.
- **Mathematical idea:** resample dependent blocks such as seasons or weeks, not iid bets, and never use the final block for selection.
- **Current implementation:** some Wilson/t-tests/bootstrap diagnostics exist, but no central candidate-aware outer protocol.
- **Validation:** publish block-level ROI/CLV intervals, P(ROI > 0), number of candidate variants, and final holdout result.

### Proposed Adaptation E — constrained portfolio Kelly (not yet implemented)

- **Technique:** uncertainty-aware fractional Kelly with exposure/correlation constraints.
- **Source:** Kelly (1956); Uhrín et al. (2021); portfolio-risk literature reviewed through the sports-betting paper.
- **Mathematical idea:** maximize conservative expected log growth subject to total stake, league/team/outcome exposure and drawdown constraints; reduce `p` toward market/base rate according to uncertainty.
- **Current implementation:** independent capped quarter-Kelly or RL multiplier.
- **Validation:** replay identical selected opportunities under flat, quarter-Kelly, constrained quarter-Kelly and full policy; compare block-bootstrap CAGR, drawdown and ruin.

## Research → Code Traceability Map

```text
Dixon & Coles (1997) / Hvattum & Arntzen (2010)
      ↓
Poisson score model + dynamic Elo
      ↓
Expected goals, score-grid 1X2 probabilities, sequential team strength
      ↓
models/poisson_elo_model.py
      ↓
Log loss/Brier/accuracy on chronological outer windows

Gneiting & Raftery (2007) + Walsh & Joshi (2023/24)
      ↓
Proper scoring and calibration-first selection
      ↓
Temporal sigmoid calibration, ECE/Brier/log loss, betting-region calibration
      ↓
models/ml_layer.py, models/calibration.py, pipeline.py
      ↓
Outer model-vs-market probability comparison and realized CLV

Kelly (1956) + Uhrín et al. (2021)
      ↓
Fractional growth-optimal sizing with risk controls
      ↓
Quarter-Kelly, caps, flat-policy control, future constrained portfolio sizing
      ↓
pipeline.py, models/rl_staking_agent.py, scripts/08_staking_stress_test.py
      ↓
Block-bootstrap drawdown, ruin and CAGR comparison

White (2000) + Bailey et al. (2014/2015) + market-efficiency-lab
      ↓
Selection-bias and backtest-overfitting control
      ↓
Nested temporal folds, candidate registry, sacred holdout, block bootstrap
      ↓
Proposed central validation engine; not yet implemented
      ↓
Probability of positive ROI and holdout robustness, not a single lucky path
```

## Implementation Roadmap

### Tier 1 — implement before any production claim

1. Add dataset snapshot IDs, hashes, odds timestamp/book columns and a manifest.
2. Centralize one nested walk-forward evaluator with a sacred final period.
3. Add market-only, model-only and market-aware blend controls.
4. Replace iid per-bet inference with season/week block bootstrap.
5. Add selected-region calibration and robust lower-edge filtering.
6. Keep fractional Kelly as default; make Q-learning challenger-only.
7. Add portfolio exposure/correlation controls and realistic execution assumptions.

### Tier 2 — research challengers

1. Bayesian/state-space dynamic attack/defense ratings.
2. Time-varying market blend with pre-registered regime rules.
3. Rich pre-match features with strict source timestamps: xG, lineup status, injuries, travel and weather.
4. Conformal/bootstrap uncertainty under a dependence-aware evaluation design.
5. Stacking with inner-fold predictions only.

### Tier 3 — avoid for now

1. Transformer/LSTM scale-up without materially more timestamp-valid data.
2. RL policy search over selection and sizing before independent validation exists.
3. Threshold/feature expansion driven by the same outer test period.

## Experimental Plan

A valid experiment table should look like this:

| Variant | Train | Inner selection | Outer evaluation | Primary acceptance criterion |
|---|---|---|---|---|
| Baseline | past only | fixed pre-registered defaults | sacred future blocks | reference |
| A: market-aware | past only | blend weight in inner temporal folds | same blocks/prices | lower market-relative log loss/Brier |
| B: uncertainty | past only | z/filter in inner folds | same blocks/prices | better selected-region calibration/CLV without unacceptable turnover |
| C: combined | past only | all choices inner only | same blocks/prices | robust block-bootstrap risk-adjusted improvement |

Do not choose a winner from ROI alone. The combined model fails the experiment if it produces higher ROI but worse calibration, negative CLV, larger drawdown or a result that disappears in one outer season.

## Adapted / Taken From Research

| Category | This project’s status |
|---|---|
| **Original work already present** | The synthetic world, repository pipeline, Poisson/Elo implementation, ML layer, adaptive/dynamic experiments, tests, ledgers, demos and reports were already in the repository before this audit. |
| **Research-inspired improvements made now** | Single-pass coherent Elo training; temporal ML diagnostic holdout; temporal sigmoid calibration folds. These are independent implementation changes motivated by the cited methodology. |
| **Direct algorithmic adaptations** | The repository already contains a Dixon–Coles-style low-score correction, Elo rating, sigmoid calibration, fractional Kelly, uncertainty sampling and walk-forward ideas. This audit did not copy source code. The new changes do not reproduce a paper’s code. |
| **Code reused** | None from external repositories. |
| **Attribution preserved** | Sources and links are listed above; repository names and licenses are listed in the GitHub review. |

## Final PhD-Level Verdict

**Is the current system genuinely competitive from a quantitative-research perspective?**

**Methodologically: yes, as a serious small research prototype. Economically: not demonstrated.** It is more defensible than a typical “AI betting predictor” because it has temporal walks, probability metrics, market comparisons, uncertainty experiments, CLV fields, negative findings, tests and reproducible synthetic data. It is not yet competitive evidence of a sustainable risk-adjusted ROI because the market-relative probability edge, CLV, realistic execution and block-level ROI inference are not established on a sufficiently large untouched real-data sample.

**What most increases the probability of robust out-of-sample ROI?**

1. Treat opening/closing market information and price availability as first-class, timestamped data, and prove incremental value beyond a market-only baseline.
2. Make nested temporal selection, block-bootstrap inference and a sacred holdout non-negotiable.
3. Use conservative calibrated probabilities and constrained fractional-Kelly portfolio sizing; do not spend the next research cycle on larger neural networks or RL.

The scientifically correct near-term objective is not “make the backtest ROI larger.” It is: **make it difficult for the system to fool itself, then see whether any edge survives.**

## Limitations of This Audit

- Web research was limited to publicly accessible source pages and repository documentation; some publisher pages were cookie/403 restricted, so DOI and bibliographic claims are included only where independently verified by accessible records.
- GitHub review was selective and methodology-focused, not an exhaustive crawl of every sports repository.
- No new real-data backtest was run during this audit; conclusions use the repository’s persisted artifacts and source code. The proposed experiments are not claims of results.
- A positive CLV diagnostic is not automatically positive ROI, and a negative short sample is not proof that no edge can ever exist.
- Betting markets, data feeds, bookmaker policies and available odds change over time; any live deployment would require a fresh, forward paper-trading record.
