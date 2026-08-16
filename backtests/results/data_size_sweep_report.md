# Data-size sweep (walk-forward, same evaluation window)

Model vs market and betting metrics per training size. Evaluation window: 300 matches after each training split.

| n_train | model LL | market LL | model Brier | market Brier | model acc | market acc | model ECE | market ECE | beats market | avg unc | bets | ROI% | adv edge% | real edge% | cal gap | CLV% | CLV t | bets(1σ) | ROI(1σ)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 1.0954 | 0.9709 | 0.6570 | 0.5771 | 0.453 | 0.550 | 0.090 | 0.045 | 0.0 | 7.60 | 143.0 | 14.68 | 78.47 | 18.57 | 19.31 | -0.12 | -0.16 | 111.0 | 29.76 |
| 200 | 1.0630 | 0.9709 | 0.6352 | 0.5771 | 0.477 | 0.550 | 0.075 | 0.045 | 0.0 | 5.03 | 133.0 | -22.27 | 55.75 | -1.82 | 21.42 | -0.28 | -0.36 | 113.0 | 17.38 |
| 400 | 1.0369 | 0.9709 | 0.6209 | 0.5771 | 0.510 | 0.550 | 0.049 | 0.045 | 0.0 | 2.89 | 134.0 | -26.23 | 53.30 | 15.83 | 15.60 | -0.80 | -1.07 | 124.0 | -33.48 |
| 600 | 1.0405 | 0.9709 | 0.6250 | 0.5771 | 0.493 | 0.550 | 0.061 | 0.045 | 0.0 | 2.15 | 123.0 | -69.56 | 49.26 | -12.07 | 23.69 | 0.13 | 0.16 | 117.0 | -69.66 |
| 780 | 1.0294 | 0.9709 | 0.6164 | 0.5771 | 0.493 | 0.550 | 0.053 | 0.045 | 0.0 | 1.80 | 104.0 | -59.66 | 41.61 | -11.47 | 20.46 | 0.39 | 0.43 | 100.0 | -60.26 |
| 1980 | 1.0063 | 0.9709 | 0.5997 | 0.5771 | 0.543 | 0.550 | 0.058 | 0.045 | 0.0 | 0.98 | 51.0 | -11.07 | 19.17 | -6.65 | 11.52 | 1.52 | 0.95 | 51.0 | -11.07 |

## Where additional data stops helping

Marginal model-log-loss change between consecutive sizes (negative = improvement): ['-0.0324', '-0.0261', '+0.0036', '-0.0111', '-0.0231']

Model log loss keeps improving with data at every step (no noise plateau reached) - sample size is still a binding constraint within this range.

## Bottleneck diagnosis

Model-vs-market log-loss gap per size (positive = model better than the market): ['-0.1245', '-0.0921', '-0.0660', '-0.0696', '-0.0585', '-0.0354']

Even at the largest training size the model is +0.0354 log loss WORSE than the market's implied probabilities. The gap narrows with data (-0.1245 -> -0.0354) but never closes, so the bottleneck is **features/model structure and market information**, not sample size alone.

## Uncertainty-adjusted edge filter

ROI raw vs ROI with edge > 1-sigma filter per size: ['14.7/29.8', '-22.3/17.4', '-26.2/-33.5', '-69.6/-69.7', '-59.7/-60.3', '-11.1/-11.1']

The 1σ filter improved ROI at 2/6 sizes (ties/NaN excluded). The uncertainty filter does not rescue the strategy: the problem is the absence of information (CLV), not merely edge overconfidence.
CLV t-stat per size: ['-0.16', '-0.36', '-1.07', '+0.16', '+0.43', '+0.95'] - information signal absent.
