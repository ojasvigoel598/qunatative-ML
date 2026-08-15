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
       La Liga                             PyTorch NN    0.4395    1.1583 0.6836
       La Liga            TF hybrid (NN + PoissonElo)    0.4237    1.3057 0.7383
       La Liga              sklearn Gradient Boosting    0.4763    1.0524 0.6345
       La Liga            sklearn Logistic Regression    0.4816    1.0604 0.6393
       La Liga               sklearn Ridge classifier    0.4816    1.0578 0.6373
       La Liga                  sklearn Random Forest    0.4789    1.0485 0.6311
       La Liga PyTorch NN - cold start (no team info)    0.4895    1.0892 0.6485
Premier League                Base rate (most common)    0.4263    1.0793 0.6534
Premier League                Market (bookmaker odds)    0.4895    1.0185 0.6115
Premier League                             PyTorch NN    0.4158    1.2614 0.7374
Premier League            TF hybrid (NN + PoissonElo)    0.3737    1.5341 0.8187
Premier League              sklearn Gradient Boosting    0.4158    1.1023 0.6695
Premier League            sklearn Logistic Regression    0.4026    1.1193 0.6808
Premier League               sklearn Ridge classifier    0.4158    1.1159 0.6780
Premier League                  sklearn Random Forest    0.4263    1.0961 0.6644
Premier League PyTorch NN - cold start (no team info)    0.4263    1.1377 0.6795
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
