#!/usr/bin/env python3
"""Quick 1M Monte Carlo test using the real pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline
from optimization.monte_carlo_engine import MonteCarloEngine

print("Generating data and running backtest...")
df = pipeline.generate_match_data(600, seed=42)
result = pipeline.run_backtest(df, use_ml=True, use_rl=False,
                               save_results=False, verbose=False)

bets_df = result["bets_df"]
print(f"Bets: {len(bets_df)}, ROI: {result['summary']['roi_pct']:+.2f}%")

if len(bets_df) > 0:
    engine = MonteCarloEngine(verbose=True)
    mc = engine.run(bets_df, initial_bankroll=10000.0,
                    n_simulations=1_000_000, seed=42)

    from pathlib import Path as P
    out_dir = ROOT / "results"
    engine.save_results(mc, out_dir / "monte_carlo_summary.json")
    engine.write_report(mc, out_dir / "monte_carlo_report.md")
    print("\n[OK] 1M Monte Carlo complete.")
else:
    print("No bets — skipping Monte Carlo.")
