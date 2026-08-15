#!/usr/bin/env python3
"""
Auto-sync watcher: re-runs notebooks/sync_notebook.py whenever a source file
changes, so the notebook (and its GitHub HTML render) always reflects the
current code.  Pure-stdlib polling — no extra dependencies.

    python notebooks/watch_sync.py                 # watch everything
    python notebooks/watch_sync.py --poll 2        # poll every 2s
    python notebooks/watch_sync.py --once          # one sync pass, then exit

Watched: models/, agent_sim/, scripts/, demo/, pipeline.py and the notebook
itself.  On any change the sync runs after a short debounce (so rapid multi-
file edits trigger one sync, not ten).  Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SYNC = ROOT / "notebooks" / "sync_notebook.py"

WATCH_DIRS = ["models", "agent_sim", "scripts", "demo"]
WATCH_FILES = ["pipeline.py", "notebooks/01_explained_ml_pipeline.ipynb"]

POLL_SECONDS = 3.0
DEBOUNCE_SECONDS = 2.0


def _watched_files() -> list[Path]:
    files: list[Path] = []
    for d in WATCH_DIRS:
        p = ROOT / d
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
    for rel in WATCH_FILES:
        p = ROOT / rel
        if p.exists():
            files.append(p)
    return files


def snapshot() -> str:
    """Hash of all watched file contents (a cheap change detector)."""
    h = hashlib.md5()
    for f in _watched_files():
        try:
            h.update(f.read_bytes())
        except OSError:
            continue
    return h.hexdigest()


def run_sync() -> int:
    env = {**os.environ, "PYTHONUTF8": "1"}
    print(f"[watch] {time.strftime('%H:%M:%S')} change detected — syncing...")
    try:
        rc = subprocess.call([str(PY), str(SYNC)], env=env, cwd=str(ROOT))
    except Exception as exc:                     # noqa: BLE001
        print(f"[watch] sync failed to start: {exc}")
        return 1
    print(f"[watch] sync finished rc={rc}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll", type=float, default=POLL_SECONDS)
    ap.add_argument("--once", action="store_true",
                    help="single sync pass, then exit")
    args = ap.parse_args()

    if args.once:
        return run_sync()

    last = snapshot()
    print(f"[watch] watching {len(_watched_files())} files "
          f"({args.poll:g}s poll). Ctrl+C to stop.")
    changed_at = None
    while True:
        time.sleep(args.poll)
        cur = snapshot()
        if cur != last:
            last = cur
            changed_at = time.time()
        elif changed_at is not None and \
                time.time() - changed_at >= DEBOUNCE_SECONDS:
            changed_at = None
            run_sync()
            last = snapshot()      # ignore the notebook the sync itself writes


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[watch] stopped.")
        sys.exit(0)
