# Real Walk-Forward Simulation — multi-season (La Liga 2021/22 .. 2025/26)

Point-in-time replay of five real La Liga seasons with a $1M start. Each
season trains on all seasons before it (expanding window) and then walks
match-by-match using ONLY information known before kick-off (online Elo/
form, real pre-match odds). Staking: quarter-Kelly capped at 5% of
bankroll, 15% daily cap, survival mode after a -90% drawdown.

```
    book  season        final       roi  n_bets  wins   strike  avg_odds  avg_edge   max_dd   avg_clv      clv_t        clv_p survival
    b365 2021/22 1.010664e+06  1.066362     100    50 0.500000  2.253200 21.072244 0.310717 -2.897303 -12.960387 4.732159e-23     None
pinnacle 2021/22 9.822123e+05 -1.778766     110    53 0.481818  2.303091 23.259244 0.326806  3.391154  15.185502 1.165460e-28     None
    b365 2022/23 1.115256e+06 11.525644      63    27 0.428571  2.416508 17.701505 0.221609 -2.661413 -10.386260 3.380305e-15     None
pinnacle 2022/23 1.255391e+06 25.539117      72    33 0.458333  2.456250 19.078808 0.212756  3.014483  12.672298 6.668067e-20     None
    b365 2023/24 9.880755e+05 -1.192447      59    26 0.440678  2.306949 13.405322 0.213215 -2.229506  -7.973932 6.772993e-11     None
pinnacle 2023/24 1.110934e+06 11.093430      67    33 0.492537  2.328060 14.715561 0.189178  2.419385   8.709428 1.429627e-12     None
    b365 2024/25 1.116514e+06 11.651430      34    15 0.441176  2.532353 13.828679 0.163941 -2.730417  -8.598881 6.148337e-10     None
pinnacle 2024/25 1.171129e+06 17.112937      40    18 0.450000  2.552500 15.319215 0.190880  2.850415   9.833748 4.104997e-12     None
    b365 2025/26 9.642495e+05 -3.575053      47    19 0.404255  2.340426 12.959547 0.159836 -2.163986  -4.902557 1.344264e-04     None
pinnacle 2025/26 9.588264e+05 -4.117364      18     7 0.388889  2.494444 10.945261 0.115625  2.538912   6.481042 5.644292e-06     None
```

## The multi-season verdict (small sample, be honest)

- Five seasons is still a small sample: the 95% CI on mean ROI is wide.
- CLV vs the sharp line (Pinnacle) is the most reliable real-world signal
  of whether the model's prices beat the market.
- Compare books: betting at B365 vs Pinnacle prices shows how much the
  soft bookmaker margin costs.

*(Saved by `demo/real_simulation.py --multi`; per-season log in
`demo/output/real_simulation_multi.csv`.)*
