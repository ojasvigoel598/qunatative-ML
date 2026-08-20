# Monte Carlo Simulation Report
**Simulations**: 1,000,000
**Bets per path**: 45
**Initial bankroll**: $10,000
**Runtime**: 10.8s

## ROI Distribution
| Statistic | Value ||-----------|------:|| Mean ROI | +194.81% || Median ROI | +156.96% || Std ROI | 165.18% || 5th percentile | +7.40% || 25th percentile | +79.54% || 75th percentile | +267.62% || 95th percentile | +510.39% |
## Risk Metrics
| Metric | Value ||--------|------:|| P(ROI > 0) | 96.3% || P(ROI > 5%) | 95.4% || P(ruin) | 0.0000 || Mean max drawdown | 26.2% || 95th %ile max drawdown | 43.2% || Mean max losing streak | 4.3 || 95th %ile losing streak | 7.0 |
## Final Bankroll Distribution
| Statistic | Value ||-----------|------:|| Mean | $29,481 || Median | $25,696 || 5th percentile | $10,740 || 95th percentile | $61,039 |
## Configuration
- outcome_noise: 0.02
- odds_noise: 0.03
- calibration_stress: 0.05
- slippage_pct: 0.01
- kelly_fraction: 0.25
- max_stake_frac: 0.08
- seed: 42
