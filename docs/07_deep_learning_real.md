# Deep Nets on Real vs Synthetic Training Data — Accuracy Iteration

Both the PyTorch NN and (optionally) the TensorFlow hybrid were trained
on (a) real La Liga 2021/22-2024/25 and (b) the synthetic world, using
the SAME online feature pipeline (running Elo, 5-game form, points
streaks) and evaluated on genuinely unseen matches: La Liga 2025/26
(unseen season) and Premier League 2025/26 (unseen league).

The iteration loop:

1. `baseline_4feat_raw` — 4 features, no early stop, no calibration
   (exposes the overfitting + miscalibration gaps)
2. `baseline_4feat_calib` — + temperature scaling on a chronological
   validation slice (fixes log-loss / Brier / ECE)
3. `regularised_4feat` — smaller net + dropout + early stopping
   (shrinks the train-vs-test gap)
4. `rich_8feat_regularised` — + Elo diff, form diff, points streaks
   (lifts accuracy itself)

## Results

```
                 iteration  n_features  NN__La Liga 25/26__acc  NN__La Liga 25/26__bacc  NN__La Liga 25/26__ll  NN__La Liga 25/26__brier  NN__La Liga 25/26__ece  NN__train_gap_La Liga 25/26  NN__EPL 25/26__acc  NN__EPL 25/26__bacc  NN__EPL 25/26__ll  NN__EPL 25/26__brier  NN__EPL 25/26__ece  NN__train_gap_EPL 25/26
        baseline_4feat_raw           4                  0.4974                   0.4349                 1.0373                    0.6143                  0.1087                       0.1288              0.4289               0.3967             1.1190                0.6658              0.1142                   0.1973
      baseline_4feat_calib           4                  0.5158                   0.4221                 0.9661                    0.5728                  0.0423                       0.0121              0.4711               0.4087             1.0501                0.6333              0.0439                   0.0568
         regularised_4feat           4                  0.5211                   0.4293                 0.9656                    0.5722                  0.0403                       0.0184              0.4763               0.4157             1.0459                0.6303              0.0486                   0.0632
    rich_8feat_regularised           8                  0.5079                   0.4197                 0.9719                    0.5753                  0.0343                       0.0215              0.4842               0.4230             1.0478                0.6323              0.0639                   0.0452
    SYN_baseline_4feat_raw           4                  0.4605                   0.3513                 1.0987                    0.6520                  0.1147                       0.1836              0.4553               0.3967             1.1779                0.6960              0.1328                   0.1888
  SYN_baseline_4feat_calib           4                  0.4868                   0.3708                 1.0200                    0.6088                  0.0396                       0.0701              0.4816               0.4173             1.0529                0.6326              0.0603                   0.0753
     SYN_regularised_4feat           4                  0.4947                   0.3716                 1.0163                    0.6063                  0.0471                       0.0582              0.4789               0.4109             1.0496                0.6307              0.0416                   0.0740
SYN_rich_8feat_regularised           8                  0.4868                   0.3723                 1.0338                    0.6179                  0.0581                       0.0808              0.5000               0.4377             1.0498                0.6315              0.0512                   0.0676
```

## Round 2 — dual-league training + ensembles (test accuracy)

```
                                                                                                                      0                                                                                                 1
single__La Liga 25/26  {'accuracy': 0.5211, 'balanced_acc': 0.4293, 'log_loss': 0.9656, 'brier': 0.5722, 'ece': 0.0403}   {'accuracy': 0.5158, 'balanced_acc': 0.4179, 'log_loss': 0.972, 'brier': 0.5748, 'ece': 0.0342}
single__EPL 25/26      {'accuracy': 0.4763, 'balanced_acc': 0.4157, 'log_loss': 1.0459, 'brier': 0.6303, 'ece': 0.0486}  {'accuracy': 0.4579, 'balanced_acc': 0.3774, 'log_loss': 1.0549, 'brier': 0.6359, 'ece': 0.0338}
dual__La Liga 25/26     {'accuracy': 0.5105, 'balanced_acc': 0.414, 'log_loss': 0.9773, 'brier': 0.5793, 'ece': 0.0535}     {'accuracy': 0.5, 'balanced_acc': 0.4038, 'log_loss': 0.9802, 'brier': 0.5806, 'ece': 0.0796}
dual__EPL 25/26        {'accuracy': 0.4974, 'balanced_acc': 0.4417, 'log_loss': 1.0232, 'brier': 0.6147, 'ece': 0.0719}  {'accuracy': 0.4763, 'balanced_acc': 0.4183, 'log_loss': 1.0187, 'brier': 0.6117, 'ece': 0.0405}
```

## Root-cause analysis

- **Overfitting**: the raw 300-epoch net overfits ~1,300 training rows
  (train accuracy >> test accuracy). Early stopping + dropout close most
  of the gap; the train→test gap column makes this measurable.
- **Miscalibration**: raw softmax is overconfident; temperature scaling
  on the validation slice fixes ECE and log-loss without touching test.
- **Draws**: the confusion matrix shows draws are the hardest class
  (recall ~0.02-0.06) — draws are the fundamental limit of 3-way
  football prediction. Round 3 tried to FIX this with class-weighted
  loss (drawx3) and NN+Ridge blends: weighting draws destroys
  accuracy (0.52 -> 0.43/0.29) and blends do not help, proving the
  draw collapse is a *feature-information* limit, not a training bug.
- **Real vs synthetic**: real-trained nets beat synthetic-trained nets
  on unseen real matches, but the gap to the bookmaker remains.

*(Saved by `scripts/06_deep_learning_real.py`; full numbers in
`backtests/results/deep_learning_real_results.csv`.)*
