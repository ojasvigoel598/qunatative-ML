# $1M Simulation with the Dynamic Thinking Layer

Every decision is made by `models/dynamic_thinking.py`: the model fuses
the trained ensemble with the **public vs sharp market split** (a hidden
signal), re-weights model-vs-market from its own rolling Brier, shrinks
stakes with model/market disagreement and drawdown, and switches to a
low-risk survival mode below 10% of the start.  The same forward match
streams are replayed under flat and Kelly staking as controls.

## Confidence-aware adaptation

The layer now adapts **in proportion to how confident it is**:

* **Confidence** = margin of the top outcome above the uniform 1/3
  (1.0 = certain, 0.0 = coin-flip).
* **Calibration blend** — the model-vs-market weight is re-weighted
  from Brier that is *weighted by confidence* (a confident wrong call
  hurts the model's trust more than a coin-flip), and the weight
  update step grows with confidence: it adapts fast on clear signals,
  cautiously when guessing.
* **Confidence-gated refits** — if rolling confidence decays > 0.08
  below its best, the base model refits on the recent window
  immediately (in addition to its scheduled/drift refits).
* **Confidence-scaled stakes** — stake x p(best)/0.40 (capped
  [0.75, 1.4]), so the layer commits more only when it is genuinely
  more sure than the minimum pass.

```
    policy         mean       median   p_profit           p5          p95        worst         best  median_cagr   avg_bets
      flat 1.365050e+06 1.361400e+06  75.000000 8.043000e+05 1.963720e+06 8.021000e+05 2.201100e+06     9.845409 643.250000
     kelly 6.397421e+06 2.720664e+05  25.000000 7.034977e+03 3.298998e+07 6.411426e+03 7.172747e+07   -32.713312 643.250000
   dynamic 1.293736e+06 1.213297e+06  91.666667 1.009954e+06 1.612846e+06 9.571032e+05 1.671870e+06     6.061431 388.666667
dynamic_v1 1.275068e+06 1.247349e+06 100.000000 1.135635e+06 1.459585e+06 1.123301e+06 1.513909e+06     6.958744 443.000000
```

**dynamic vs its own v1 baseline (fixed base, no fusion, no confidence):** 
median $1,213,297 vs $1,247,349, 
P(profit) 92% vs 100%, 
90% range [$1,009,954 .. $1,612,846] vs 
[$1,135,635 .. $1,459,585].  Final model-vs-market weight 
0.48, base refits/trial 13.5 (confidence-gated 
1.5), final rolling confidence 0.27.

**Honest reading:** the dynamic layer adapts its reasoning (final market
weight, uncertainty shrink, drawdown risk, survival switch) instead of
applying a fixed rule — but in this synthetic world with a modest edge
the *flat* policy's variance control is hard to beat.  The value of the
thinking layer shows on real data where the sharp line carries genuine
information (see `docs/09_real_walkforward_simulation.md` CLV results).

*(Saved by `demo/simulation_dynamic.py`; per-trial numbers in
`demo/output/simulation_1m_dynamic.csv`.)*
