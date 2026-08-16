# Deep-Learning Transfer Experiment

Two deep models - a **PyTorch MLP** and a **TensorFlow hybrid** (an MLP
fused with the PoissonElo model's probability outputs) - were trained on
**all 1,200 synthetic matches** (seed 42) and then evaluated on *real*
match data they had never seen:

| Test set | Question |
|----------|----------|
| La Liga 2025/26 (SP1) | Cross-league: does the learned feature→probability mapping transfer to a different league? |
| Premier League 2025/26 (E0) | Out-of-sample: does it work on real matches in the same league the synthetic data mimics? |

Features for the real leagues are computed from the **previous** real season
(2024/25): Elo ratings and shifted rolling form. A cold-start row (no team
information at all) shows the null result.

## Results

```
        league                                 method  accuracy  log_loss  brier
       La Liga                Base rate (most common)    0.4895    1.0464 0.6299
       La Liga                Market (bookmaker odds)    0.5447    0.9637 0.5710
       La Liga                             PyTorch NN    0.4158    1.1481 0.6878
       La Liga            TF hybrid (NN + PoissonElo)    0.4711    1.2369 0.7187
       La Liga              sklearn Gradient Boosting    0.4553    1.0738 0.6493
       La Liga            sklearn Logistic Regression    0.4684    1.0666 0.6440
       La Liga               sklearn Ridge classifier    0.4737    1.0615 0.6410
       La Liga                  sklearn Random Forest    0.4342    1.0807 0.6550
       La Liga PyTorch NN - cold start (no team info)    0.4895    1.0639 0.6384
Premier League                Base rate (most common)    0.4263    1.0793 0.6534
Premier League                Market (bookmaker odds)    0.4895    1.0185 0.6115
Premier League                             PyTorch NN    0.3921    1.2442 0.7358
Premier League            TF hybrid (NN + PoissonElo)    0.4132    1.3646 0.7723
Premier League              sklearn Gradient Boosting    0.4079    1.1420 0.6942
Premier League            sklearn Logistic Regression    0.4184    1.1296 0.6877
Premier League               sklearn Ridge classifier    0.4105    1.1267 0.6854
Premier League                  sklearn Random Forest    0.3684    1.1449 0.6983
Premier League PyTorch NN - cold start (no team info)    0.4263    1.1062 0.6671
```

*(Saved by `scripts/04_deep_learning_transfer.py`; full numbers in
`backtests/results/transfer_results.csv`.)*

## What this demonstrates

- If the models beat the base rate but trail the market, the synthetic-trained
  mapping transfers *partially*: it learned something real about football, but
  the real bookmaker remains the strongest predictor.
- The cold-start row quantifies how much of the accuracy comes from team
  information vs prior probabilities alone.
- The conclusion for real deployment: retrain on real data (see
  `scripts/01_data_ingestion.py`); the synthetic world validates the
  methodology, not the absolute numbers.
