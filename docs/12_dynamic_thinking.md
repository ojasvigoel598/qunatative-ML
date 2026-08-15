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
    policy          mean       median  p_profit            p5          p95         worst         best  median_cagr  avg_bets
      flat 943996.000000 938000.00000      32.0 759900.000000 1.148660e+06 689100.000000 1.245600e+06    -1.929308    127.88
     kelly 842895.864698 762548.85587      32.0 417673.922202 1.540433e+06 393694.013005 1.586256e+06    -7.920019    127.88
   dynamic 965757.353268 964813.51430      44.0 869225.253500 1.048257e+06 796876.446900 1.064738e+06    -1.084363    118.68
dynamic_v1 967327.862412 955929.54090      28.0 904552.561380 1.035677e+06 889144.062800 1.122468e+06    -1.362484     57.92
```

**dynamic vs its own v1 baseline (fixed base, no fusion, no confidence):** 
median $964,814 vs $955,930, 
P(profit) 44% vs 28%, 
90% range [$869,225 .. $1,048,257] vs 
[$904,553 .. $1,035,677].  Final model-vs-market weight 
0.50, base refits/trial 14.5 (confidence-gated 
2.5), final rolling confidence 0.30.

**Honest reading:** the dynamic layer adapts its reasoning (final market
weight, uncertainty shrink, drawdown risk, survival switch) instead of
applying a fixed rule — but in this synthetic world with a modest edge
the *flat* policy's variance control is hard to beat.  The value of the
thinking layer shows on real data where the sharp line carries genuine
information (see `docs/09_real_walkforward_simulation.md` CLV results).

*(Saved by `demo/simulation_dynamic.py`; per-trial numbers in
`demo/output/simulation_1m_dynamic.csv`.)*
