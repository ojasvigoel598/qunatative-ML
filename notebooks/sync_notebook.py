#!/usr/bin/env python3
"""
Auto-sync the explained ML notebook after every code change.

The notebook's live sections (the randomised 100-seed multi-league aggregate
and the ATP tennis walk-forward) are re-generated from the LATEST result CSVs
on every sync, so the notebook always shows what the current code produced.

    python notebooks/sync_notebook.py              # refresh sections + execute + validate + HTML
    python notebooks/sync_notebook.py --no-html    # skip the HTML render
    python notebooks/sync_notebook.py --check      # validate an already-executed notebook only

The companion watcher (notebooks/watch_sync.py) calls this automatically
whenever any source file changes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "01_explained_ml_pipeline.ipynb"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
KERNEL_DIR = ROOT / ".venv" / "share" / "jupyter"
HTML_PATH = ROOT / "notebooks" / "01_explained_ml_pipeline.html"

CONCLUSION_MARK = "## 22. Reproducibility"
SENTINEL_AGG = "# SYNC-SECTION: agent_sim_aggregate"
SENTINEL_TENNIS = "# SYNC-SECTION: tennis_walkforward"

# --------------------------------------------------------------------------
# Live section 1 — randomised 100-seed multi-league aggregate
# --------------------------------------------------------------------------
AGG_MD = """## 22. Randomised multi-league walk-forward — 100-seed aggregate

Section 19 walked ONE randomised world. One run proves nothing — the honest
question is the DISTRIBUTION over many independent scenarios. `scripts/13_multi_league_agent.py`
ran the same strictly chronological, leakage-audited engine across **100 fresh seeds**;
each run drew a random league subset (La Liga / EPL / Bundesliga / Serie A), a random
walk season, a random start date and a random league-reveal order — so the agent could
never know in advance which league would be profitable. Every table lives in CSVs under
`backtests/results/agent_sim/` (nothing is printed to the terminal). The cells below read
those CSVs live, so they always show the latest run.
"""

AGG_CODE = f"""# {SENTINEL_AGG}
# Read the live aggregate written by scripts/13_multi_league_agent.py
agg = pd.read_csv(results_dir / "agent_sim" / "multi_run_aggregate.csv")
runs = pd.read_csv(results_dir / "agent_sim" / "multi_run_runs.csv")
freq = pd.read_csv(results_dir / "agent_sim" / "league_selection_frequency.csv")

a = agg.iloc[0]
print(f"Across {{int(a['n_runs'])}} independent runs:")
print(f"  mean ROI {{a['mean_roi']:+.2f}}%   median {{a['median_roi']:+.2f}}%   "
      f"std {{a['std_roi']:.2f}}%   P(profit) {{a['pct_profitable']:.0f}}%")
print(f"  final bankroll  mean ${{a['mean_final_bankroll']:,.0f}}   "
      f"median ${{a['median_final_bankroll']:,.0f}}")
print(f"  worst ${{a['worst_final']:,.0f}}   best ${{a['best_final']:,.0f}}")
print(f"  mean max drawdown {{a['mean_max_drawdown']:.1%}}   "
      f"bets/run {{a['mean_bets_per_run']:.1f}}   matches/run {{a['mean_matches_evaluated']:.0f}}")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
axes[0].hist(runs["roi_pct"], bins=15, color="#2e86ab", edgecolor="white")
axes[0].axvline(0, color="crimson", ls="--", lw=1.5)
axes[0].set_title(f"ROI distribution across {{int(a['n_runs'])}} runs "
                  f"(mean {{a['mean_roi']:+.1f}}%)")
axes[0].set_xlabel("ROI on staked (%)")
axes[1].bar(freq["league_code"], freq["selection_freq_pct"], color="#f6ae2d")
axes[1].set_title("League selection frequency")
axes[1].set_xlabel("League code"); axes[1].set_ylabel("% of runs selecting it")
plt.tight_layout(); plt.show()"""

# --------------------------------------------------------------------------
# Live section 2 — different sport: ATP tennis walk-forward
# --------------------------------------------------------------------------
TENNIS_MD = """## 23. Different sport — ATP tennis walk-forward (no future knowledge)

Everything above is FOOTBALL (3-outcome). Does the methodology transfer to a genuinely
different sport? Tennis is the stress test: **2-outcome matches, no draws**, a surface
structure, and a different data schema. `scripts/14_tennis_walkforward.py` re-applies the
same disciplines — an ADAPTIVE online Elo that only ever sees matches before the current
moment, a strict chronological walk with a leakage audit, edge-based staking at the B365
public price, Pinnacle as the sharp CLV reference, survival mode, and baselines — on real
ATP 2016-2025 data with real odds, fetched on demand from the free tennis-data.co.uk
archive (no key). The walk years (2024, 2025) were never touched by training.
"""

TENNIS_CODE = f"""# {SENTINEL_TENNIS}
# Read the live walk-forward results written by scripts/14_tennis_walkforward.py
ten = pd.read_csv(results_dir / "tennis_walkforward" / "tennis_summary.csv")
print(ten[["walk_year", "mode", "bets", "win_rate", "accuracy", "brier",
           "roi_pct", "mean_clv_pct", "max_drawdown"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for y in sorted(ten["walk_year"].unique()):
    sub = ten[ten["walk_year"] == y]
    axes[0].bar(sub["mode"] + " " + str(y), sub["roi_pct"],
                label=str(y), color="#5f0f40" if y == min(ten["walk_year"]) else "#e36414")
axes[0].axhline(0, color="crimson", ls="--", lw=1.2)
axes[0].set_title("Tennis walk-forward ROI by policy")
axes[0].set_ylabel("ROI on staked (%)")
axes[0].tick_params(axis="x", rotation=30)
ml2024 = ten[(ten["mode"] == "ml") & (ten["walk_year"] == 2024)]
if len(ml2024):
    axes[1].bar(["ML 2024", "Implied 2024", "Random 2024"],
                [ml2024["mean_clv_pct"].iloc[0],
                 ten[(ten["mode"] == "implied") & (ten["walk_year"] == 2024)]["mean_clv_pct"].iloc[0],
                 ten[(ten["mode"] == "random") & (ten["walk_year"] == 2024)]["mean_clv_pct"].iloc[0]],
                color=["#2e86ab", "#f6ae2d", "#5f0f40"])
axes[1].axhline(0, color="crimson", ls="--", lw=1.2)
axes[1].set_title("CLV vs Pinnacle (sharp line) — ATP 2024")
axes[1].set_ylabel("Mean CLV (%)")
plt.tight_layout(); plt.show()

# Per-surface breakdown for the ML policy
surf = pd.read_csv(results_dir / "tennis_walkforward" / "tennis_by_surface_2024.csv")
print("\\nML policy — profit by court surface (ATP 2024):")
print(surf.to_string(index=False))"""


# --------------------------------------------------------------------------
def refresh_live_sections(nb) -> int:
    """Insert (if missing) or refresh (in place) the two live sections before
    the conclusion cell.  Returns how many sections were touched."""
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell

    cells = nb["cells"]
    # locate the conclusion cell (last occurrence of the mark)
    concl_idx = next((i for i, c in enumerate(cells)
                      if c["cell_type"] == "markdown"
                      and CONCLUSION_MARK in c["source"]),
                     len(cells))

    def upsert(sentinel, md, code):
        code_idx = next((i for i, c in enumerate(cells)
                         if c["cell_type"] == "code"
                         and sentinel in c["source"]), None)
        touched = 0
        if code_idx is not None:
            cells[code_idx]["source"] = code       # refresh live code in place
            touched += 1
        else:
            cells.insert(concl_idx, new_code_cell(code))
            cells.insert(concl_idx, new_markdown_cell(md))
            touched += 1
        return touched

    n = 0
    n += upsert(SENTINEL_AGG, AGG_MD, AGG_CODE)
    n += upsert(SENTINEL_TENNIS, TENNIS_MD, TENNIS_CODE)
    return n


def execute_notebook(timeout: int = 900) -> int:
    env = {"JUPYTER_PATH": str(KERNEL_DIR), "PYTHONUTF8": "1"}
    cmd = [str(PY), "-m", "nbconvert", "--to", "notebook",
           "--execute", "--inplace", str(NB_PATH),
           f"--ExecutePreprocessor.timeout={timeout}",
           "--ExecutePreprocessor.allow_errors=False"]
    return subprocess.call(cmd, env={**__import__("os").environ, **env})


def check_errors() -> int:
    import nbformat
    nb = nbformat.read(NB_PATH, as_version=4)
    errs = [c for c in nb["cells"] if c["cell_type"] == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errs:
        print(f"[sync] {len(errs)} cell error(s):")
        for e in errs[:5]:
            ename = e.get("ename", "?")
            evalue = str(e.get("evalue", ""))[:200]
            print(f"   {ename}: {evalue}")
    return len(errs)


def render_html() -> int:
    env = {"JUPYTER_PATH": str(KERNEL_DIR), "PYTHONUTF8": "1"}
    cmd = [str(PY), "-m", "nbconvert", "--to", "html",
           "--template", "classic", "--embed-images",
           "--output", HTML_PATH.name, str(NB_PATH)]
    return subprocess.call(cmd, cwd=str(HTML_PATH.parent),
                           env={**__import__("os").environ, **env})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-html", action="store_true",
                    help="skip the HTML render for GitHub")
    ap.add_argument("--check", action="store_true",
                    help="only validate the notebook, do not execute")
    args = ap.parse_args()

    if not NB_PATH.exists():
        print(f"[sync] notebook not found: {NB_PATH}", file=sys.stderr)
        return 1

    if not args.check:
        import nbformat
        nb = nbformat.read(NB_PATH, as_version=4)
        touched = refresh_live_sections(nb)
        with open(NB_PATH, "w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        print(f"[sync] live sections refreshed ({touched} touched)")
        rc = execute_notebook()
        if rc != 0:
            print(f"[sync] execution failed (rc={rc})", file=sys.stderr)
            return rc
        n_err = check_errors()
        if n_err:
            print(f"[sync] {n_err} cell error(s) — fix before committing",
                  file=sys.stderr)
            return 2
        print("[sync] notebook executed, 0 errors")
        if not args.no_html:
            rc = render_html()
            print(f"[sync] HTML rendered: {HTML_PATH.name} "
                  f"({HTML_PATH.stat().st_size / 1024:.0f} KB)" if rc == 0
                  else f"[sync] HTML render failed rc={rc}")
            return rc
    else:
        n_err = check_errors()
        print(f"[sync] check: {n_err} error(s)")
        return n_err
    return 0


if __name__ == "__main__":
    sys.exit(main())
