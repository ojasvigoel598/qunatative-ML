#!/usr/bin/env python3
"""
End-of-run reports and multi-run aggregation, written to CSV (req 10, 11).

Tables are NOT printed to the terminal — they are saved under
backtests/results/agent_sim/ as <run_id>_report.csv and multi_run_aggregate.csv.
The terminal only prints one-line summaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "backtests" / "results" \
    / "agent_sim"

PERF_STEP = 50   # equity sample every N resolved matches for "performance over time"


def build_run_report(run_id: str, seed: int, world, engine, ledger) -> pd.DataFrame:
    """One row with every end-of-run metric (req 10) + league breakdown."""
    rows = list(ledger.rows)
    bets = [r for r in rows if r["decision"] is not None and not r["invalidated"]]
    no_bets = [r for r in rows if r["decision"] is None]
    wins = [r for r in bets if r["profit"] and r["profit"] > 0]

    profits = [r["profit"] for r in bets]
    staked = sum(r["stake"] for r in bets)
    final = bets[-1]["bankroll_after"] if bets else agent_start(ledger)
    start = bets[0]["bankroll_before"] if bets else agent_start(ledger)

    # max drawdown from bankroll path
    peak, max_dd = -np.inf, 0.0
    for r in bets:
        b = r["bankroll_after"]
        peak = max(peak, b)
        max_dd = max(max_dd, (peak - b) / peak if peak else 0.0)

    # performance over time
    perf = [{"match_idx": i, "date": r["kickoff"], "bankroll": r["bankroll_after"]}
            for i, r in enumerate(bets) if i % PERF_STEP == 0]
    perf_df = pd.DataFrame(perf)
    if len(perf_df):
        perf_df.to_csv(RESULTS_DIR / f"{run_id}_equity.csv", index=False)

    # league breakdown
    lg_rows = []
    for r in rows:
        if r["decision"] is not None and not r["invalidated"]:
            lg_rows.append(r)
    leagues = {}
    for lg_code in set(r["league_code"] for r in lg_rows):
        sub = [r for r in lg_rows if r["league_code"] == lg_code]
        lg_profits = sum(r["profit"] for r in sub)
        lg_staked = sum(r["stake"] for r in sub)
        leagues[lg_code] = {
            "league": sub[0]["league"], "n_bets": len(sub),
            "wins": sum(1 for r in sub if r["profit"] > 0),
            "profit": round(lg_profits, 2),
            "roi_pct": round(lg_profits / lg_staked * 100, 2) if lg_staked else 0.0,
            "staked": round(lg_staked, 2),
        }
    league_df = pd.DataFrame(leagues).T.reset_index().rename(
        columns={"index": "league_code"})
    if len(league_df):
        # stable row order: dict keys come from set() iteration, which varies
        # with the hash seed between runs (breaks byte-identical reproducibility)
        league_df = league_df.sort_values("league_code").reset_index(drop=True)
        league_df.to_csv(RESULTS_DIR / f"{run_id}_by_league.csv", index=False)

    # league availability + selection timeline
    timeline = engine.league_timeline()
    if len(timeline):
        timeline.to_csv(RESULTS_DIR / f"{run_id}_league_timeline.csv", index=False)

    report = {
        "run_id": run_id, "seed": seed,
        "season": world.walk_season,
        "start_date": world.start_date, "end_date": world.end_date,
        "leagues_encountered": "|".join(world.leagues),
        "leagues_analysed": "|".join(sorted({r["league_code"] for r in rows})),
        "matches_evaluated": len(rows),
        "bets_placed": len(bets), "no_bets": len(no_bets),
        "wins": len(wins), "losses": len(bets) - len(wins),
        "win_rate": round(len(wins) / len(bets), 4) if bets else 0.0,
        "start_bankroll": round(start, 2),
        "final_bankroll": round(final, 2),
        "total_profit": round(sum(profits), 2),
        "total_staked": round(staked, 2),
        "roi_pct": round(sum(profits) / staked * 100, 2) if staked else 0.0,
        "avg_odds": round(float(np.mean([_bet_odds(r) for r in bets])) if bets else 0.0, 3),
        "avg_model_prob": round(float(np.mean([max(r["prob_home"], r["prob_draw"], r["prob_away"]) for r in bets])) if bets else 0.0, 3),
        "avg_edge": round(float(np.mean([r["edge"] for r in bets])) if bets else 0.0, 4),
        "max_drawdown": round(max_dd, 4),
        "n_leak_flags": engine.n_leak_flags,
        "frozen_window": engine.frozen_armed,
        "survival": agent_summary(engine).get("survival", False),
        "model_refits": agent_summary(engine).get("model_refits", 0),
    }
    report_df = pd.DataFrame([report])
    report_df.to_csv(RESULTS_DIR / f"{run_id}_report.csv", index=False)
    return report_df


def _bet_odds(row: dict) -> float:
    """The odds of the outcome that was actually bet (not the home odds)."""
    d = row.get("decision")
    if d == "home_win":
        return float(row.get("odds_home") or 0.0)
    if d == "draw":
        return float(row.get("odds_draw") or 0.0)
    if d == "away_win":
        return float(row.get("odds_away") or 0.0)
    return 0.0


def agent_start(ledger) -> float:
    """Starting bankroll (first recorded bankroll_before)."""
    for r in ledger.rows:
        if r["bankroll_before"] is not None:
            return r["bankroll_before"]
    return 1_000_000.0


def agent_summary(engine):
    s = getattr(engine.agent, "summary", None)
    return s() if s else {}


def aggregate_runs(reports: list[dict], seeds: list[int]) -> pd.DataFrame:
    """Cross-run statistics (req 11) + league-selection frequency."""
    df = pd.DataFrame(reports)
    if not len(df):
        return pd.DataFrame()
    agg = {
        "n_runs": len(df),
        "mean_roi": round(df["roi_pct"].mean(), 2),
        "median_roi": round(df["roi_pct"].median(), 2),
        "std_roi": round(df["roi_pct"].std(), 2),
        "min_roi": round(df["roi_pct"].min(), 2),
        "max_roi": round(df["roi_pct"].max(), 2),
        "mean_final_bankroll": round(df["final_bankroll"].mean(), 2),
        "median_final_bankroll": round(df["final_bankroll"].median(), 2),
        "pct_profitable": round((df["total_profit"] > 0).mean() * 100, 1),
        "worst_final": round(df["final_bankroll"].min(), 2),
        "best_final": round(df["final_bankroll"].max(), 2),
        "mean_max_drawdown": round(df["max_drawdown"].mean(), 4),
        "median_max_drawdown": round(df["max_drawdown"].median(), 4),
        "mean_bets_per_run": round(df["bets_placed"].mean(), 1),
        "mean_matches_evaluated": round(df["matches_evaluated"].mean(), 1),
    }
    agg_df = pd.DataFrame([agg])
    agg_df.to_csv(RESULTS_DIR / "multi_run_aggregate.csv", index=False)

    # league-selection frequency
    freq = {}
    for r in df.itertuples():
        for lg in str(getattr(r, "leagues_analysed", "")).split("|"):
            if lg:
                freq[lg] = freq.get(lg, 0) + 1
    if freq:
        freq_df = pd.DataFrame(
            [{"league_code": k, "runs_selected": v,
              "selection_freq_pct": round(v / len(df) * 100, 1)}
             for k, v in sorted(freq.items())])
        freq_df.to_csv(RESULTS_DIR / "league_selection_frequency.csv", index=False)

    # per-run report (the rows used for the aggregate)
    df.to_csv(RESULTS_DIR / "multi_run_runs.csv", index=False)
    return agg_df
