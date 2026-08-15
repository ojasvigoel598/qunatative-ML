#!/usr/bin/env python3
"""
Compatibility entry point.

`run_full_project_with_ml_rl.py` is an alias for `run_full_ml_rl.py` - both run
the complete PoissonElo + ML + RL pipeline.  Kept so older references keep
working.

Usage:
    python run_full_project_with_ml_rl.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_full_ml_rl import main  # noqa: E402

if __name__ == "__main__":
    main()
