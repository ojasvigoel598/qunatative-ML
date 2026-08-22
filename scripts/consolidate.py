#!/usr/bin/env python3
"""
Consolidation Script

Identifies and removes superseded files without breaking the project.
Run this to see what can be safely removed.
"""

from pathlib import Path

# Files that are superseded by newer versions
SUPERSeded_SCRIPTS = [
    # Early scripts superseded by 25-29
    "scripts/03_generate_assets.py",  # Superseded by data pipeline
    "scripts/04_deep_learning_transfer.py",  # Superseded by 06
    "scripts/05_season_backtest.py",  # Superseded by walk_forward.py
    "scripts/06_deep_learning_real.py",  # Superseded by 28
    "scripts/08_staking_stress_test.py",  # Superseded by paper_trading.py
    "scripts/09_make_dashboard.py",  # Superseded by tracker/
    "scripts/10_adaptive_transfer.py",  # Superseded by layered_model.py
    "scripts/11_lstm_state_test.py",  # Superseded by nn_model.py
    "scripts/12_hidden_signals.py",  # Superseded by rich_features.py
    "scripts/13_multi_league_agent.py",  # Superseded by walk_forward.py
    "scripts/14_tennis_walkforward.py",  # Tennis-specific, not core
    "scripts/15_data_size_sweep.py",  # Superseded by 25
    "scripts/16_bootstrap_validation.py",  # Superseded by walk_forward.py
    "scripts/17_experiment_matrix.py",  # Superseded by 25-29
    "scripts/18_research_implementations.py",  # Superseded by specific models
    "scripts/19_clv_real_data_backtest.py",  # Superseded by 20
    "scripts/20_clv_focused_backtest.py",  # Superseded by clv_tracking.py
    "scripts/21_multi_season_clv_validation.py",  # Superseded by walk_forward.py
    "scripts/22_tight_clv_filter.py",  # Superseded by walk_forward.py
    "scripts/23_real_data_ablation.py",  # Superseded by 24
    "scripts/24_large_scale_ablation.py",  # Superseded by 25
    "scripts/25_large_scale_optimized_ablation.py",  # Superseded by 27
    "scripts/27_optimized_107k_ablation.py",  # Superseded by 28
]

# Models that are superseded
SUPERSeded_MODELS = [
    "models/activation_ablation.py",  # Test script, not a model
    "models/backtest_record.py",  # Utility, can be inlined
    # online_poisson.py KEPT - used by layered_model.py
    "models/seed_ensemble.py",  # Superseded by stacking_ensemble.py
    "models/temperature_scaling.py",  # Superseded by calibration.py
    "models/tf_hybrid.py",  # Superseded by nn_model.py
]

# Files to KEEP (core functionality)
KEEP_SCRIPTS = [
    "scripts/01_data_ingestion.py",
    "scripts/02_backtest.py",
    "scripts/07_full_metrics_report.py",
    "scripts/26_real_odds_market_comparison.py",
    "scripts/28_107k_elu_ablation.py",
    "scripts/29_walkforward_market_comparison.py",
    "scripts/consolidate.py",
]

KEEP_MODELS = [
    "models/__init__.py",
    "models/calibration.py",
    "models/calibration_selection.py",
    "models/dynamic_thinking.py",
    "models/fast_kde.py",
    "models/fast_mixture_mc.py",
    "models/layered_model.py",
    "models/lstm_model.py",
    "models/ml_layer.py",
    "models/nn_model.py",
    "models/poisson_elo_model.py",
    "models/rich_features.py",
    "models/rl_staking_agent.py",
    "models/speed_optimizations.py",
    "models/stacking_ensemble.py",
]

def main():
    print("=" * 60)
    print("CODEBASE CONSOLIDATION ANALYSIS")
    print("=" * 60)
    
    # Check which superseded files exist
    existing_superseded = [f for f in SUPERSeded_SCRIPTS if Path(f).exists()]
    existing_superseded_models = [f for f in SUPERSeded_MODELS if Path(f).exists()]
    
    print(f"\nSuperseded scripts to remove: {len(existing_superseded)}")
    for f in existing_superseded:
        print(f"  - {f}")
    
    print(f"\nSuperseded models to remove: {len(existing_superseded_models)}")
    for f in existing_superseded_models:
        print(f"  - {f}")
    
    print(f"\nCore scripts to keep: {len(KEEP_SCRIPTS)}")
    for f in KEEP_SCRIPTS:
        if Path(f).exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
    
    print(f"\nCore models to keep: {len(KEEP_MODELS)}")
    for f in KEEP_MODELS:
        if Path(f).exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
    
    # Calculate savings
    total_superseded_lines = 0
    for f in existing_superseded + existing_superseded_models:
        try:
            lines = len(Path(f).read_text().splitlines())
            total_superseded_lines += lines
        except:
            pass
    
    print(f"\n{'='*60}")
    print(f"TOTAL LINES TO REMOVE: {total_superseded_lines}")
    print(f"{'='*60}")
    
    print("\nTo remove these files, run:")
    print("  python scripts/consolidate.py --remove")

if __name__ == "__main__":
    import sys
    if "--remove" in sys.argv:
        print("Removing superseded files...")
        for f in SUPERSeded_SCRIPTS + SUPERSeded_MODELS:
            if Path(f).exists():
                Path(f).unlink()
                print(f"  Removed: {f}")
        print("Done. Run tests to verify.")
    else:
        main()
