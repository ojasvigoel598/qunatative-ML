# Season-by-Season Backtest on Real League Data

Time-aware validation on **real** match data (football-data.co.uk). Every test
prediction is a genuinely unseen future match; features for match *i* use only
matches strictly before *i* (running Elo + last-5 form).

## Within-league (La Liga, expanding window)

```
                                    experiment               method  accuracy  balanced_acc  log_loss  brier    ece
La Liga: train 2021/22-2021/22 -> test 2022/23 Majority / base rate    0.4789        0.3333    1.0595 0.6390 0.0447
La Liga: train 2021/22-2021/22 -> test 2022/23     PoissonElo model    0.5211        0.4331    1.0202 0.6079 0.0568
La Liga: train 2021/22-2021/22 -> test 2022/23     Ridge classifier    0.5316        0.4387    1.0228 0.6078 0.0467
La Liga: train 2021/22-2021/22 -> test 2022/23    Gradient Boosting    0.5026        0.3824    1.0348 0.6215 0.0599
La Liga: train 2021/22-2021/22 -> test 2022/23        Random Forest    0.5263        0.4271    1.0189 0.6105 0.0613
La Liga: train 2021/22-2022/23 -> test 2023/24 Majority / base rate    0.4395        0.3333    1.0753 0.6504 0.0171
La Liga: train 2021/22-2022/23 -> test 2023/24     PoissonElo model    0.5447        0.4798    0.9649 0.5763 0.0709
La Liga: train 2021/22-2022/23 -> test 2023/24     Ridge classifier    0.5447        0.4821    0.9588 0.5723 0.0528
La Liga: train 2021/22-2022/23 -> test 2023/24    Gradient Boosting    0.4921        0.4077    1.0121 0.6062 0.0316
La Liga: train 2021/22-2022/23 -> test 2023/24        Random Forest    0.5395        0.4723    0.9871 0.5875 0.0449
La Liga: train 2021/22-2023/24 -> test 2024/25 Majority / base rate    0.4447        0.3333    1.0712 0.6477 0.0061
La Liga: train 2021/22-2023/24 -> test 2024/25     PoissonElo model    0.5421        0.4644    0.9809 0.5832 0.0957
La Liga: train 2021/22-2023/24 -> test 2024/25     Ridge classifier    0.5500        0.4765    0.9814 0.5827 0.0795
La Liga: train 2021/22-2023/24 -> test 2024/25    Gradient Boosting    0.5158        0.4389    0.9983 0.5958 0.0343
La Liga: train 2021/22-2023/24 -> test 2024/25        Random Forest    0.5316        0.4619    0.9913 0.5903 0.0240
La Liga: train 2021/22-2024/25 -> test 2025/26 Majority / base rate    0.4895        0.3333    1.0496 0.6323 0.0401
La Liga: train 2021/22-2024/25 -> test 2025/26     PoissonElo model    0.5342        0.4287    0.9828 0.5819 0.0561
La Liga: train 2021/22-2024/25 -> test 2025/26     Ridge classifier    0.5158        0.4176    0.9809 0.5792 0.0481
La Liga: train 2021/22-2024/25 -> test 2025/26    Gradient Boosting    0.5184        0.4182    0.9790 0.5816 0.0459
La Liga: train 2021/22-2024/25 -> test 2025/26        Random Forest    0.5184        0.4296    0.9814 0.5832 0.0444
```

## Cross-league transfer

```
                                          experiment               method  accuracy  balanced_acc  log_loss  brier    ece
La Liga 21/22-24/25 -> Premier League 25/26 (unseen) Majority / base rate    0.4263        0.3333    1.0804 0.6542 0.0230
La Liga 21/22-24/25 -> Premier League 25/26 (unseen)     PoissonElo model    0.4658        0.3818    1.0511 0.6338 0.0363
La Liga 21/22-24/25 -> Premier League 25/26 (unseen)     Ridge classifier    0.4737        0.3955    1.0394 0.6267 0.0303
La Liga 21/22-24/25 -> Premier League 25/26 (unseen)    Gradient Boosting    0.4763        0.4186    1.0518 0.6333 0.0443
La Liga 21/22-24/25 -> Premier League 25/26 (unseen)        Random Forest    0.4737        0.4165    1.0531 0.6347 0.0349
Premier League 21/22-24/25 -> La Liga 25/26 (unseen) Majority / base rate    0.4895        0.3333    1.0546 0.6355 0.0441
Premier League 21/22-24/25 -> La Liga 25/26 (unseen)     PoissonElo model    0.4868        0.3783    1.0291 0.6167 0.0711
Premier League 21/22-24/25 -> La Liga 25/26 (unseen)     Ridge classifier    0.4895        0.3906    1.0258 0.6130 0.0496
Premier League 21/22-24/25 -> La Liga 25/26 (unseen)    Gradient Boosting    0.4868        0.3798    1.0202 0.6096 0.0606
Premier League 21/22-24/25 -> La Liga 25/26 (unseen)        Random Forest    0.5026        0.4026    1.0139 0.6055 0.0605
```

*(Saved by `scripts/05_season_backtest.py`; full numbers in
`backtests/results/season_backtest_results.csv`.)*
