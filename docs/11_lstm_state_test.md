# Stateful Sequence Models (LSTM / GRU) on Real Data

An LSTM/GRU over each team's rolling last-8 match sequence — the hidden
state IS the team's learned evolving form — compared with the feed-forward
baselines under the identical point-in-time protocol (train strictly before
the test window, predict -> reveal -> update online, zero future info).

## Head-to-head on unseen 2025/26 matches (trained on Serie A 2020/21-2023/24)

```
                                    experiment                   method  accuracy  balanced_acc  log_loss  brier
      Serie A -> Serie A 25/26 (within-league)     Majority / base rate    0.3895        0.3333    1.0869 0.6592
      Serie A -> Serie A 25/26 (within-league)         PoissonElo model    0.5184        0.4638    1.0049 0.6008
      Serie A -> Serie A 25/26 (within-league)        Gradient Boosting    0.5026        0.4512    1.0250 0.6142
      Serie A -> Serie A 25/26 (within-league) Adaptive (online refits)    0.4921        0.4444    1.0200 0.6109
      Serie A -> Serie A 25/26 (within-league)              LSTM (rich)    0.5053        0.4613    1.0255 0.6151
      Serie A -> Serie A 25/26 (within-league)               GRU (rich)    0.4921        0.4407    1.0211 0.6133
      Serie A -> Serie A 25/26 (within-league)   LSTM thin (goals only)    0.5053        0.4522    1.0135 0.6074
      Serie A -> Serie A 25/26 (within-league)       LSTM rich, NO odds    0.5053        0.4613    1.0255 0.6151
       Serie A -> La Liga 25/26 (cross-league)     Majority / base rate    0.4895        0.3333    1.0599 0.6399
       Serie A -> La Liga 25/26 (cross-league)         PoissonElo model    0.5079        0.3911    1.0315 0.6184
       Serie A -> La Liga 25/26 (cross-league)        Gradient Boosting    0.4974        0.3978    1.0306 0.6163
       Serie A -> La Liga 25/26 (cross-league) Adaptive (online refits)    0.4658        0.3671    1.0372 0.6221
       Serie A -> La Liga 25/26 (cross-league)              LSTM (rich)    0.4789        0.4056    1.0371 0.6205
       Serie A -> La Liga 25/26 (cross-league)               GRU (rich)    0.4868        0.3988    1.0476 0.6295
       Serie A -> La Liga 25/26 (cross-league)   LSTM thin (goals only)    0.4816        0.3823    1.0197 0.6092
       Serie A -> La Liga 25/26 (cross-league)       LSTM rich, NO odds    0.4789        0.4056    1.0371 0.6205
Serie A -> Premier League 25/26 (cross-league)     Majority / base rate    0.4263        0.3333    1.0805 0.6542
Serie A -> Premier League 25/26 (cross-league)         PoissonElo model    0.4658        0.3902    1.0473 0.6314
Serie A -> Premier League 25/26 (cross-league)        Gradient Boosting    0.4684        0.4052    1.0581 0.6381
Serie A -> Premier League 25/26 (cross-league) Adaptive (online refits)    0.4447        0.3893    1.0707 0.6469
Serie A -> Premier League 25/26 (cross-league)              LSTM (rich)    0.4632        0.4210    1.0448 0.6316
Serie A -> Premier League 25/26 (cross-league)               GRU (rich)    0.4789        0.4282    1.0519 0.6351
Serie A -> Premier League 25/26 (cross-league)   LSTM thin (goals only)    0.4842        0.4237    1.0454 0.6309
Serie A -> Premier League 25/26 (cross-league)       LSTM rich, NO odds    0.4632        0.4210    1.0448 0.6316
```

## Database vs model (your thesis, tested directly)

The SAME LSTM is trained on 1, 2 and 3 real leagues; the test is Serie A
2025/26 in every row. If the database matters more than the architecture,
accuracy should rise with database size.

```
                                       experiment      method  accuracy  balanced_acc  log_loss  brier
DB-size: 1 league (Serie A only) -> Serie A 25/26 LSTM (rich)    0.5000        0.4490    1.0208 0.6125
  DB-size: 2 leagues (+ La Liga) -> Serie A 25/26 LSTM (rich)    0.4921        0.4415    1.0194 0.6106
      DB-size: 3 leagues (+ EPL) -> Serie A 25/26 LSTM (rich)    0.5053        0.4525    1.0140 0.6072
```

*(Saved by `scripts/11_lstm_state_test.py`; full numbers in
`backtests/results/lstm_state_results.csv`.)*
