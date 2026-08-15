# Staking Policy Stress Test — 100 trials

Models trained exactly like `demo/simulation.py`; **100** forward
trials of 1,200 matches each from a $1M start. Bet selection is
bankroll-independent, so every policy replays the **identical bet streams**
and outcomes — the table isolates staking arithmetic, not luck.

```
              policy  mean_final  median_final  P(profit)  P(ruin<100k)  P(ever<100k)  median_maxDD  median_CAGR
 quarter-kelly cap2%   1851883.0     1398638.0       0.67          0.00          0.00         0.491       0.1075
 quarter-kelly cap5%   4015739.0      665156.0       0.41          0.15          0.22         0.864      -0.1167
quarter-kelly cap10%   9200040.0       80323.0       0.20          0.54          0.74         0.981      -0.5358
    half-kelly cap5%   3940548.0      820048.0       0.43          0.12          0.23         0.867      -0.0586
    full-kelly cap5%   3984036.0      840865.0       0.44          0.13          0.23         0.867      -0.0514
   tenth-kelly cap2%   1763248.0     1247795.0       0.66          0.00          0.00         0.494       0.0697
           flat $10k   1308121.0     1311700.0       0.77          0.00          0.01         0.245       0.0861
            flat $5k   1154060.0     1155850.0       0.77          0.00          0.00         0.137       0.0451
```

## Reading

- Ruin probability rises sharply with the stake cap: quarter-Kelly at 10%
  cap shows more variance than at 5%; the flat $10k policy (≈1% per bet,
  no compounding) has the lowest ruin but also the lowest median growth.
- The best risk-adjusted policy is the one whose P(ever < $100K) stays
  low while median CAGR stays positive — check the table for the winner.
- A single policy's mean is pulled up by the fat tail; median and
  P(ruin) are the honest summary.

*(Saved by `scripts/08_staking_stress_test.py`.)*
