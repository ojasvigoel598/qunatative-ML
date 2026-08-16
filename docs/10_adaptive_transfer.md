# Adaptive Cross-League / Cross-Sport Transfer

One model is trained on **real Serie A 2020/21-2023/24** and then pointed at
unseen matches in Serie A, La Liga, the Premier League (all 2025/26) and a
synthetic basketball-like league. Every prediction uses only information
known before kick-off; after each match the online state (Elo, form) updates
and the **adaptive** model optionally refits its ML layer on a rolling window
when scheduled or when its rolling Brier drifts. The **static** control never
refits, so the gap isolates the value of adaptation.

```
                                                experiment             method  majority_acc  accuracy  balanced_acc  log_loss  brier  acc_first_half  acc_second_half  refits  base_acc
    Serie A -> Serie A 25/26 (within-league unseen season)           Adaptive      0.389474    0.4974        0.4554    1.0285 0.6166          0.4895           0.5053      32  0.389474
    Serie A -> Serie A 25/26 (within-league unseen season) Static (frozen ML)      0.389474    0.4868        0.4350    1.0189 0.6098          0.4737           0.5000       0  0.389474
          Serie A -> La Liga 25/26 (cross-league transfer)           Adaptive      0.489474    0.4632        0.3608    1.0474 0.6289          0.4895           0.4368      28  0.489474
          Serie A -> La Liga 25/26 (cross-league transfer) Static (frozen ML)      0.489474    0.4868        0.3542    1.0412 0.6261          0.4737           0.5000       0  0.489474
   Serie A -> Premier League 25/26 (cross-league transfer)           Adaptive      0.426316    0.4395        0.3806    1.0640 0.6428          0.4947           0.3842      32  0.426316
   Serie A -> Premier League 25/26 (cross-league transfer) Static (frozen ML)      0.426316    0.4474        0.3637    1.0646 0.6429          0.4684           0.4263       0  0.426316
Football -> synthetic basketball-like league (cross-sport)           Adaptive      0.548750    0.8063        0.7996    0.7457 0.4196          0.7775           0.8350      60  0.548750
Football -> synthetic basketball-like league (cross-sport) Static (frozen ML)      0.548750    0.8113        0.8022    0.9146 0.5366          0.7850           0.8375       0  0.548750
```

*(Saved by `scripts/10_adaptive_transfer.py`; full numbers in
`backtests/results/adaptive_transfer_results.csv`.)*
