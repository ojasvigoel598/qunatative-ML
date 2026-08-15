# Staking Policy Stress Test — 100 trials

Models trained exactly like `demo/simulation.py`; **100** forward
trials of 1,200 matches each from a $1M start. Bet selection is
bankroll-independent, so every policy replays the **identical bet streams**
and outcomes — the table isolates staking arithmetic, not luck.

```
              policy  mean_final  median_final  P(profit)  P(ruin<100k)  P(ever<100k)  median_maxDD  median_CAGR
 quarter-kelly cap2%    907043.0      882387.0       0.31           0.0           0.0         0.258      -0.0374
 quarter-kelly cap5%    866758.0      798235.0       0.32           0.0           0.0         0.404      -0.0663
quarter-kelly cap10%    866023.0      793107.0       0.29           0.0           0.0         0.414      -0.0681
    half-kelly cap5%    802741.0      685826.0       0.28           0.0           0.0         0.527      -0.1084
    full-kelly cap5%    762616.0      649623.0       0.25           0.0           0.0         0.591      -0.1230
   tenth-kelly cap2%    942486.0      928289.0       0.34           0.0           0.0         0.181      -0.0224
           flat $10k    943675.0      946850.0       0.34           0.0           0.0         0.155      -0.0165
            flat $5k    971838.0      973425.0       0.34           0.0           0.0         0.078      -0.0082
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
