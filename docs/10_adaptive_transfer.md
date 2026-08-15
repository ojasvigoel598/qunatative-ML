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
    Serie A -> Serie A 25/26 (within-league unseen season)           Adaptive      0.389474    0.5000        0.4568    1.0260 0.6154          0.4842           0.5158      32  0.389474
    Serie A -> Serie A 25/26 (within-league unseen season) Static (frozen ML)      0.389474    0.4816        0.4299    1.0198 0.6103          0.4632           0.5000       0  0.389474
          Serie A -> La Liga 25/26 (cross-league transfer)           Adaptive      0.489474    0.4500        0.3449    1.0426 0.6261          0.4737           0.4263      28  0.489474
          Serie A -> La Liga 25/26 (cross-league transfer) Static (frozen ML)      0.489474    0.4816        0.3476    1.0410 0.6260          0.4632           0.5000       0  0.489474
   Serie A -> Premier League 25/26 (cross-league transfer)           Adaptive      0.426316    0.4263        0.3663    1.0692 0.6462          0.4684           0.3842      32  0.426316
   Serie A -> Premier League 25/26 (cross-league transfer) Static (frozen ML)      0.426316    0.4421        0.3587    1.0631 0.6418          0.4684           0.4158       0  0.426316
Football -> synthetic basketball-like league (cross-sport)           Adaptive      0.548750    0.7825        0.7748    0.7455 0.4220          0.7600           0.8050      64  0.548750
Football -> synthetic basketball-like league (cross-sport) Static (frozen ML)      0.548750    0.8075        0.7980    0.9147 0.5379          0.7725           0.8425       0  0.548750
```

*(Saved by `scripts/10_adaptive_transfer.py`; full numbers in
`backtests/results/adaptive_transfer_results.csv`.)*
