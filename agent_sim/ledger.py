#!/usr/bin/env python3
"""
Rolling CSV ledger for the agent simulation.

The user-facing tables are NOT printed to the terminal — they live in CSVs
under backtests/results/agent_sim/:

  * agent_sim_live.csv       — the LIVE ledger: rewritten every 20 resolved
                               matches with ONLY the most recent 20 rows
                               (older rows are automatically dropped, so the
                               file stays small).  Refreshed throughout the run.
  * <run_id>_opportunities.csv — EVERY match evaluated (bets AND no-bets) with
                               the full audit fields (req 8).
  * <run_id>_bets.csv        — the complete betting transaction table (req 9).
  * <run_id>_nobets.csv      — every no-bet opportunity, with the reason.
  * <run_id>_report.csv      — the end-of-run report (req 10).
  * multi_run_aggregate.csv  — cross-run statistics (req 11) + league-selection
                               frequency.

Every opportunity row records the leakage audit fields: prediction timestamp,
data cutoff (max date of data the agent had), kickoff, and a leak flag —
if a leak is ever detected the bet is INVALIDATED (excluded from bankroll).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "backtests" / "results" \
    / "agent_sim"

OPP_COLS = [
    "bet_no", "timestamp", "kickoff", "data_cutoff", "league", "league_code",
    "match", "home", "away",
    "prediction", "prob_home", "prob_draw", "prob_away",
    "odds_home", "odds_draw", "odds_away", "sharp_home", "sharp_draw",
    "sharp_away", "edge", "threshold", "confidence", "rest_home", "rest_away",
    "league_roi", "league_seen",
    "decision", "reason", "stake", "result", "result_known_at_prediction",
    "leak_flag", "invalidated", "bankroll_before", "profit", "bankroll_after",
]

LIVE_ROWS = 20          # the live CSV keeps only these most recent rows
REFRESH_EVERY = 20      # rewrite the live CSV every N resolved matches


class RollingLedger:
    """Writes the live rolling CSV during the run + full artifacts at the end."""

    def __init__(self, run_id: str, results_dir: Path = RESULTS_DIR):
        self.run_id = run_id
        self.dir = results_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.n_resolved = 0

    # ------------------------------------------------------------ recording
    def add(self, row: dict):
        """Record one opportunity (bet or no-bet), with audit fields."""
        self.rows.append(row)

    def note_resolved(self):
        """Call after a match result is revealed: refresh the live CSV."""
        self.n_resolved += 1
        if self.n_resolved % REFRESH_EVERY == 0:
            self._write_live()

    # ------------------------------------------------------------ live CSV
    def _write_live(self):
        recent = self.rows[-LIVE_ROWS:]
        df = pd.DataFrame(recent, columns=OPP_COLS)
        df.to_csv(self.dir / "agent_sim_live.csv", index=False)
        # keep only the latest LIVE_ROWS in memory view used by the live file
        return df

    def live_path(self) -> Path:
        return self.dir / "agent_sim_live.csv"

    # ------------------------------------------------------------ artifacts
    def save_all(self):
        """Persist the full per-run artifacts."""
        if not self.rows:
            self._write_live()
            return
        df = pd.DataFrame(self.rows, columns=OPP_COLS)
        df.to_csv(self.dir / f"{self.run_id}_opportunities.csv", index=False)

        bets = df[df["decision"].notna()].copy()
        if len(bets):
            bets.to_csv(self.dir / f"{self.run_id}_bets.csv", index=False)
        else:
            pd.DataFrame(columns=OPP_COLS).to_csv(
                self.dir / f"{self.run_id}_bets.csv", index=False)

        nobets = df[df["decision"].isna()].copy()
        nobets.to_csv(self.dir / f"{self.run_id}_nobets.csv", index=False)
        self._write_live()   # final live snapshot (latest LIVE_ROWS rows)
