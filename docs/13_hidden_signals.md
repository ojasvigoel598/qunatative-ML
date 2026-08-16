# Hidden Signals on Real Data

The cached football-data.co.uk CSVs carry up to a dozen bookmakers per
match (B365, Bwin, Betfair, Betvictor, Bet365, Pinnacle, Max/Avg).  This
experiment tests the 'hidden signals' thesis on real Serie A 2025/26
matches with point-in-time knowledge only:

1. **Consensus & dispersion** — do matches where the market disagrees
   resist prediction?
2. **Public vs sharp split** — betting the public (B365) price when the
   sharp (Pinnacle) line disagrees: does it beat the closing market (CLV)?
3. **The DynamicThinkingLayer** — the fully self-refitting model fused
   with all real signals, walked over 2025/26.

## Dispersion vs predictability

Consensus accuracy on low-dispersion matches: **52.1%**
Consensus accuracy on high-dispersion matches: **56.3%**

## Sharp-vs-public split CLV

n=200 · avg CLV **+1.16%** (t=4.72, p=0.0) · positive 94/200

## Dynamic layer on real 2025/26

Final **$932,412** · ROI **-6.8%** · bets 17 · strike 17.65% · online refits 2

*(Saved by `scripts/12_hidden_signals.py`; full numbers in
`backtests/results/hidden_signals_results.csv`.)*
