# Refactor Plan

## Issues Found

### 1. Duplicate Calibration Functions
- `calibration.py` and `calibration_selection.py` both have:
  - `expected_calibration_error()`
  - `brier_score()`
  - `log_loss()`
- **Fix:** Merge `calibration_selection.py` into `calibration.py`

### 2. Oversized Files
- `pipeline.py` (784 lines) — main pipeline
- `analysis/research_layer.py` (686 lines) — research analysis
- `models/layered_model.py` (680 lines) — layered model
- `models/poisson_elo_model.py` (505 lines) — Poisson Elo
- **Fix:** Keep as-is if cohesive; split only if genuinely helps

### 3. Dead Code
- `models/fast_kde.py:benchmark_kde()` — never called
- `models/fast_mixture_mc.py:benchmark_mixture_mc()` — never called
- `models/online_poisson.py:benchmark_online_poisson()` — never called
- **Fix:** Remove benchmark functions

### 4. Wrapper Functions
- `models/dynamic_thinking.py:implied_probs()` — duplicates `calibration.py:implied_probs()`
- **Fix:** Remove duplicate, import from calibration

## Refactor Order

1. Merge calibration_selection.py into calibration.py
2. Remove duplicate implied_probs from dynamic_thinking.py
3. Remove unused benchmark functions
4. Update all imports
5. Run tests
