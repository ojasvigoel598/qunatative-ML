# Data-size sweep (walk-forward, same evaluation window)

Model vs market and betting metrics per training size. Evaluation window: 300 matches after each training split.

| n_train | model LL | market LL | model Brier | market Brier | model acc | market acc | model ECE | market ECE | beats market | avg unc | bets | ROI% | adv edge% | real edge% | cal gap | CLV% | CLV t | bets(1σ) | ROI(1σ)% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 1.0954 | 0.9709 | 0.6570 | 0.5764 | 0.453 | 0.553 | 0.090 | 0.049 | 0.0 | 7.60 | 140.0 | -7.57 | 75.48 | 14.68 | 19.79 | 1.89 | 2.98 | 108.0 | -0.95 |
| 200 | 1.0630 | 0.9709 | 0.6352 | 0.5764 | 0.477 | 0.553 | 0.075 | 0.049 | 0.0 | 5.03 | 130.0 | -31.89 | 53.08 | -0.82 | 20.47 | 1.68 | 2.54 | 109.0 | -11.45 |
| 400 | 1.0369 | 0.9709 | 0.6209 | 0.5764 | 0.510 | 0.553 | 0.049 | 0.049 | 0.0 | 2.89 | 130.0 | -41.32 | 50.92 | 11.89 | 16.29 | 1.21 | 1.91 | 119.0 | -48.11 |
| 600 | 1.0405 | 0.9709 | 0.6250 | 0.5764 | 0.493 | 0.553 | 0.061 | 0.049 | 0.0 | 2.15 | 121.0 | -72.71 | 45.99 | -16.44 | 24.38 | 2.24 | 3.22 | 112.0 | -71.95 |
| 780 | 1.0294 | 0.9709 | 0.6164 | 0.5764 | 0.493 | 0.553 | 0.053 | 0.049 | 0.0 | 1.80 | 101.0 | -64.86 | 39.15 | -12.71 | 20.37 | 2.33 | 3.32 | 98.0 | -66.48 |
| 1980 | 1.0063 | 0.9709 | 0.5997 | 0.5764 | 0.543 | 0.553 | 0.058 | 0.049 | 0.0 | 0.98 | 48.0 | -17.44 | 16.37 | -16.58 | 14.79 | 3.47 | 3.03 | 48.0 | -17.44 |

## Where additional data stops helping

Marginal model-log-loss change between consecutive sizes (negative = improvement): ['-0.0324', '-0.0261', '+0.0036', '-0.0111', '-0.0231']

Model log loss keeps improving with data at every step (no noise plateau reached) - sample size is still a binding constraint within this range.

## Bottleneck diagnosis

Model-vs-market log-loss gap per size (positive = model better than the market): ['-0.1245', '-0.0921', '-0.0660', '-0.0696', '-0.0585', '-0.0354']

Even at the largest training size the model is +0.0354 log loss WORSE than the market's implied probabilities. The gap narrows with data (-0.1245 -> -0.0354) but never closes, so the bottleneck is **features/model structure and market information**, not sample size alone.

## Uncertainty-adjusted edge filter

ROI raw vs ROI with edge > 1-sigma filter per size: ['-7.6/-0.9', '-31.9/-11.4', '-41.3/-48.1', '-72.7/-72.0', '-64.9/-66.5', '-17.4/-17.4']

The 1σ filter improved ROI at 3/6 sizes (ties/NaN excluded). The uncertainty filter helps: raw edges are partially overconfidence.
CLV t-stat per size: ['+2.98', '+2.54', '+1.91', '+3.22', '+3.32', '+3.03'] - information signal present.

Note: significant positive CLV coexists with negative ROI. The model beats the closing line on PRICE but still loses, which isolates the failure in the PROBABILITIES (the betting-region calibration gap), not in price selection.
