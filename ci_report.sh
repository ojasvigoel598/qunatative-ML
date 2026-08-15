#!/usr/bin/env bash
# =============================================================================
# One-command report pipeline: regenerate EVERY artifact, then verify.
#
#   ./ci_report.sh            # full run (deep-learning experiments included)
#   ./ci_report.sh --fast     # skip the long deep-learning experiments (04, 06)
#
# Runs, in order:
#   1. canonical backtests (PoissonElo + Kelly, and + ML + RL)
#   2. $1M flat-staking simulation
#   3. real walk-forward replay (multi-season, both books)
#   4. deep-learning transfer experiment  (skipped with --fast)
#   5. real-data season backtest
#   6. deep nets real-vs-synthetic iteration (skipped with --fast)
#   7. full metrics report + staking stress test + interactive dashboard
#   8. verification suite
#
# Requires: Python venv at .venv/ (see README Installation).
# On Windows use Git Bash:  bash ci_report.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"          # POSIX venv layout
[ -x "$PY" ] || { echo "ERROR: venv not found (run README installation)"; exit 1; }

FAST=0
if [ "${1:-}" = "--fast" ]; then FAST=1; shift; fi
if [ "${CI_SKIP_HEAVY:-0}" = "1" ]; then FAST=1; fi
# CI_SMOKE=1 runs ONLY the report + test steps (fast CI verification)
SMOKE=0
if [ "${CI_SMOKE:-0}" = "1" ]; then SMOKE=1; fi

step() { echo; echo "============================================================"; echo "  $1"; echo "============================================================"; }

set +e
if [ "$SMOKE" = "0" ]; then
step "[1/8] Canonical backtests"
"$PY" run_full_project.py;  echo "  -> base:      $?"
"$PY" run_full_ml_rl.py;    echo "  -> ML+RL:     $?"
else
step "[1/8] Canonical backtests - SKIPPED (CI_SMOKE)"
fi

if [ "$SMOKE" = "0" ]; then
step "[2/8] \$1M simulation (flat staking, variance-minimising)"
"$PY" demo/simulation.py --trials 25 --matches 1200

step "[3/8] Real walk-forward replay (multi-season, both books)"
"$PY" demo/real_simulation.py --multi --offline

if [ "$FAST" = "0" ]; then
    step "[4/8] Deep-learning transfer (PyTorch NN + TF hybrid) - takes a few minutes"
    "$PY" scripts/04_deep_learning_transfer.py --offline
else
    step "[4/8] Deep-learning transfer - SKIPPED (--fast)"
fi

step "[5/8] Real-data season-by-season backtest"
"$PY" scripts/05_season_backtest.py --offline

if [ "$FAST" = "0" ]; then
    step "[6/8] Deep nets on real vs synthetic data (iteration loop) - takes a few minutes"
    "$PY" scripts/06_deep_learning_real.py --offline
else
    step "[6/8] Deep nets real-vs-synthetic - SKIPPED (--fast)"
fi
else
step "[2/8..6/8] heavy experiments - SKIPPED (CI_SMOKE)"
fi

step "[7/8] Reports: metrics + staking stress test + interactive dashboard"
"$PY" scripts/07_full_metrics_report.py
"$PY" scripts/08_staking_stress_test.py --trials 100
"$PY" scripts/09_make_dashboard.py

step "[8/8] Verification suite"
"$PY" -m pytest tests/ -q
PYTEST_RC=$?
set -e

echo
echo "============================================================"
echo "  DONE - artifacts regenerated under:"
echo "    backtests/results/   demo/output/   docs/dashboard.html"
echo "  pytest exit code: $PYTEST_RC"
echo "============================================================"
exit $PYTEST_RC
