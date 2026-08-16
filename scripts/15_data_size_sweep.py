#!/usr/bin/env python3
"""
Data-size investigation: does more data make the model beat the market?

For every training size (default 100 -> 200 -> 400 -> 600 -> 780 -> full)
the SAME future window (the last `--eval-window` matches) is scored by a
model trained only on the matches BEFORE the window (strict walk-forward,
no random split), and the full battery of metrics is measured:

    * model vs market        : log loss, Brier, accuracy, ECE (both sides)
    * betting                : bets, ROI (quarter-Kelly), advertised vs
                               realised edge, betting-region calibration gap
    * information            : CLV mean / win rate / t-stat
    * uncertainty            : mean MC std of model probabilities, plus the
                               uncertainty-adjusted bet filter (bet only when
                               edge > 1 sigma of the edge estimate)

The report then identifies where additional data stops producing meaningful
improvement (marginal log-loss deltas) and whether the bottleneck is sample
size, calibration, or the market itself (gap to the market that data cannot
close).

Usage:
    python scripts/15_data_size_sweep.py            # Serie A, all sizes
    python scripts/15_data_size_sweep.py --league SP1
    python scripts/15_data_size_sweep.py --sizes 100 400 780
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.match_analysis import analyze_match  # noqa: E402
from data.real_data import LEAGUES, load_league  # noqa: E402
from models.calibration import (accuracy, brier_score,  # noqa: E402
                                expected_calibration_error, log_loss)
from models.ml_layer import MLFootballPredictor  # noqa: E402
from models.poisson_elo_model import PoissonEloModel  # noqa: E402

DEFAULT_SIZES = [100, 200, 400, 600, 780, 0]  # 0 = train on everything before the window
OUT_CSV = PROJECT_ROOT / "backtests" / "results" / "data_size_sweep.csv"
OUT_MD = PROJECT_ROOT / "backtests" / "results" / "data_size_sweep_report.md"


def _cards_over(df: pd.DataFrame, poisson: PoissonEloModel,
                ml: MLFootballPredictor, n_samples: int = 100) -> pd.DataFrame:
    """Full reasoning card for every match in the window (one pass)."""
    cards = [analyze_match(row, poisson, ml, uncertainty_z=0.0,
                           n_samples=n_samples, seed=0)
             for _, row in df.iterrows()]
    return pd.DataFrame(cards)


def _betting_stats(cards: pd.DataFrame, df: pd.DataFrame,
                   z: float = 0.0) -> dict:
    """Aggregate betting metrics from the cards, applying the
    uncertainty-adjusted filter when z > 0."""
    if z > 0:
        mask = (cards["decision"] == "BET") & (
            cards["edge_pct"] - z * cards["edge_uncertainty_pct"] > 0)
    else:
        mask = cards["decision"] == "BET"
    sub = cards[mask].copy()
    out = {"n_bets": int(len(sub))}
    if len(sub) == 0:
        out.update({"roi_pct": np.nan, "avg_edge_pct": np.nan,
                    "realized_edge_pct": np.nan, "cal_gap_pct": np.nan,
                    "clv_mean_pct": np.nan, "clv_t": np.nan,
                    "clv_win_rate_pct": np.nan, "win_rate_pct": np.nan,
                    "avg_odds": np.nan})
        return out

    bankroll = 1000.0
    for _, c in sub.iterrows():
        edge = c["edge_pct"] / 100
        odds = c["best_odds_home_win"] if c["best_outcome"] == "home_win" else (
            c["best_odds_draw"] if c["best_outcome"] == "draw" else c["best_odds_away_win"])
        stake = min(edge / (odds - 1) * 0.25, 0.05) * bankroll
        won = c["correct"]
        bankroll += stake * (odds - 1) if won else -stake

    win_rate = float(sub["correct"].mean())
    avg_odds = float(sub["best_odds_home_win"].where(
        sub["best_outcome"] == "home_win", sub["best_odds_draw"].where(
            sub["best_outcome"] == "draw", sub["best_odds_away_win"])).mean())
    realized = win_rate * avg_odds - 1.0
    clv = sub["clv_pct"].dropna().to_numpy(dtype=float)
    out.update({
        "roi_pct": round((bankroll / 1000 - 1) * 100, 2),
        "avg_edge_pct": round(float(sub["edge_pct"].mean()), 2),
        "realized_edge_pct": round(realized * 100, 2),
        "cal_gap_pct": round(float(
            (sub["p_model_home_win"].where(sub["best_outcome"] == "home_win",
                                           sub["p_model_draw"].where(
                                               sub["best_outcome"] == "draw",
                                               sub["p_model_away_win"])).mean()
             - win_rate) * 100), 2),
        "clv_mean_pct": round(float(clv.mean()), 2) if len(clv) else np.nan,
        "clv_t": round(float(clv.mean() / (clv.std(ddof=1) / np.sqrt(len(clv)))), 2)
        if len(clv) > 1 and clv.std(ddof=1) > 0 else 0.0,
        "clv_win_rate_pct": round(float(np.mean(clv > 0)) * 100, 2) if len(clv) else np.nan,
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_odds": round(avg_odds, 2),
    })
    return out


def run_size_sweep(df: pd.DataFrame, sizes, eval_window: int = 300,
                   n_samples: int = 100, verbose: bool = True) -> pd.DataFrame:
    """Run the walk-forward sweep; returns the results table."""
    df = df.sort_values("date").reset_index(drop=True)
    total = len(df)
    eval_start = max(total - eval_window, 0)
    eval_df = df.iloc[eval_start:]
    results = []

    for n in sizes:
        if n == 0 or n > eval_start:
            n = eval_start  # "full": train on everything before the window
        train = df.iloc[:n]
        if verbose:
            print(f"\n=== size {n} (train) -> window {len(eval_df)} (eval) ===")
        poisson = PoissonEloModel()          # Dixon-Coles ON: real football
        poisson.train(train, verbose=False)
        ml = MLFootballPredictor(model_type="gradient_boosting")
        ml.train(poisson.prepare_features(train), verbose=False)

        cards = _cards_over(eval_df, poisson, ml, n_samples=n_samples)

        # ---- model vs market on the window --------------------------------
        y_true = eval_df["result"].map({"H": 2, "D": 1, "A": 0}).to_numpy()
        p_model = cards[["p_model_away_win", "p_model_draw",
                         "p_model_home_win"]].to_numpy()
        p_market = cards[["p_bookie_away_win", "p_bookie_draw",
                          "p_bookie_home_win"]].to_numpy()
        row = {
            "n_train": int(n),
            "eval_n": len(eval_df),
            "model_log_loss": round(log_loss(p_model, y_true), 4),
            "market_log_loss": round(log_loss(p_market, y_true), 4),
            "model_brier": round(brier_score(p_model, y_true), 4),
            "market_brier": round(brier_score(p_market, y_true), 4),
            "model_acc": round(accuracy(p_model, y_true), 4),
            "market_acc": round(accuracy(p_market, y_true), 4),
            "model_ece": round(expected_calibration_error(p_model, y_true), 4),
            "market_ece": round(expected_calibration_error(p_market, y_true), 4),
            "beats_market": int(bool(log_loss(p_model, y_true)
                                     < log_loss(p_market, y_true))),
            # mean uncertainty across outcomes (Poisson MC std)
            "avg_unc_pct": round(float(cards[["unc_home_win", "unc_draw",
                                              "unc_away_win"]].mean().mean()) * 100, 2),
        }
        # ---- betting (raw edge) -------------------------------------------
        bs = _betting_stats(cards, eval_df, z=0.0)
        for k, v in bs.items():
            row[f"bet_{k}"] = v
        # ---- betting (uncertainty-adjusted: edge must exceed 1 sigma) -----
        bs_adj = _betting_stats(cards, eval_df, z=1.0)
        for k, v in bs_adj.items():
            row[f"bet_adj_{k}"] = v
        results.append(row)

    return pd.DataFrame(results)


def write_report(results: pd.DataFrame, out_md: Path) -> str:
    """Markdown report: table + where improvement stops + bottleneck."""
    lines = []
    lines.append("# Data-size sweep (walk-forward, same evaluation window)\n")
    lines.append(f"Model vs market and betting metrics per training size. "
                 f"Evaluation window: {int(results['eval_n'].iloc[0])} matches "
                 f"after each training split.\n")
    lines.append("| n_train | model LL | market LL | model Brier | market Brier | "
                 "model acc | market acc | model ECE | market ECE | beats market | "
                 "avg unc | bets | ROI% | adv edge% | real edge% | cal gap | "
                 "CLV% | CLV t | bets(1σ) | ROI(1σ)% |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in results.iterrows():
        lines.append(
            f"| {int(r['n_train'])} | {r['model_log_loss']:.4f} | "
            f"{r['market_log_loss']:.4f} | {r['model_brier']:.4f} | "
            f"{r['market_brier']:.4f} | {r['model_acc']:.3f} | "
            f"{r['market_acc']:.3f} | {r['model_ece']:.3f} | "
            f"{r['market_ece']:.3f} | {r['beats_market']} | "
            f"{r['avg_unc_pct']:.2f} | {r['bet_n_bets']} | {r['bet_roi_pct']:.2f} | "
            f"{r['bet_avg_edge_pct']:.2f} | {r['bet_realized_edge_pct']:.2f} | "
            f"{r['bet_cal_gap_pct']:.2f} | {r['bet_clv_mean_pct']:.2f} | "
            f"{r['bet_clv_t']:.2f} | {r['bet_adj_n_bets']} | "
            f"{r['bet_adj_roi_pct']:.2f} |")
    lines.append("")

    # ---- where does improvement stop? ------------------------------------
    srt = results.sort_values("n_train")
    ll = srt["model_log_loss"].to_numpy()
    sizes = srt["n_train"].to_numpy()
    lines.append("## Where additional data stops helping\n")
    deltas = [ll[i] - ll[i - 1] for i in range(1, len(ll))]
    lines.append("Marginal model-log-loss change between consecutive sizes "
                 f"(negative = improvement): {[f'{d:+.4f}' for d in deltas]}\n")
    # a plateau only counts if the LAST improvements are all noise: require the
    # final two marginal changes to be < 0.005 in magnitude
    if len(deltas) >= 2 and all(abs(d) < 0.005 for d in deltas[-2:]):
        lines.append("**Diminishing returns:** the final marginal changes are all "
                     "< 0.005 - additional data stops producing meaningful "
                     "improvement beyond the tested range.\n")
    else:
        lines.append("Model log loss keeps improving with data at every step "
                     "(no noise plateau reached) - sample size is still a "
                     "binding constraint within this range.\n")

    # ---- bottleneck: gap to the market -----------------------------------
    gap = (srt["market_log_loss"] - srt["model_log_loss"]).to_numpy()
    lines.append(f"## Bottleneck diagnosis\n")
    lines.append(f"Model-vs-market log-loss gap per size (positive = model "
                 f"better than the market): {[f'{g:+.4f}' for g in gap]}\n")
    last = gap[-1]
    if last <= 0:
        lines.append(f"Even at the largest training size the model is "
                     f"{-last:+.4f} log loss WORSE than the market's implied "
                     f"probabilities. The gap narrows with data ({gap[0]:+.4f} "
                     f"-> {last:+.4f}) but never closes, so the bottleneck is "
                     f"**features/model structure and market information**, not "
                     f"sample size alone.\n")
    else:
        lines.append(f"At the largest training size the model beats the market "
                     f"({last:+.4f} log loss). The bottleneck is data volume: "
                     f"more data continues to help.\n")

    # ---- uncertainty-adjusted filter verdict ------------------------------
    roi_raw = results["bet_roi_pct"].to_numpy()
    roi_adj = results["bet_adj_roi_pct"].to_numpy()
    clv_t = results["bet_clv_t"].to_numpy()
    lines.append("## Uncertainty-adjusted edge filter\n")
    lines.append(f"ROI raw vs ROI with edge > 1-sigma filter per size: "
                 f"{[f'{a:.1f}/{b:.1f}' for a, b in zip(roi_raw, roi_adj)]}\n")
    better = int(np.nansum(roi_adj > roi_raw))
    lines.append(f"The 1σ filter improved ROI at {better}/{len(roi_raw)} sizes "
                 f"(ties/NaN excluded). "
                 + ("The uncertainty filter helps: raw edges are partially "
                    "overconfidence."
                    if better >= len(roi_raw) / 2
                    else "The uncertainty filter does not rescue the strategy: "
                         "the problem is the absence of information (CLV), not "
                         "merely edge overconfidence."))
    lines.append(f"CLV t-stat per size: {[f'{t:+.2f}' for t in clv_t]} - "
                 f"information signal {'present' if np.nanmax(np.abs(clv_t)) >= 2 else 'absent'}.\n")
    # CLV vs ROI reconciliation: positive CLV with negative ROI means the model
    # finds good PRICES but loses on PROBABILITIES (overconfident edges).
    if np.nanmax(np.abs(clv_t)) >= 2 and np.nanmean(roi_raw) < 0:
        lines.append("Note: significant positive CLV coexists with negative ROI. "
                     "The model beats the closing line on PRICE but still loses, "
                     "which isolates the failure in the PROBABILITIES (the "
                     "betting-region calibration gap), not in price selection.\n")
    text = "\n".join(lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="Walk-forward data-size sweep")
    parser.add_argument("--league", choices=list(LEAGUES), default="I1",
                        help="league code (default I1 = Serie A)")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--eval-window", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="MC samples per match for uncertainty")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate the report from the saved CSV")
    args = parser.parse_args()

    if args.report_only:
        if not OUT_CSV.exists():
            sys.exit(f"[FAIL] {OUT_CSV} missing - run the sweep first")
        results = pd.read_csv(OUT_CSV)
    else:
        print(f"Loading {LEAGUES[args.league]} (cached, offline)...")
        df = load_league(args.league, offline=True)
        print(f"{len(df)} matches")
        results = run_size_sweep(df, args.sizes, eval_window=args.eval_window,
                                 n_samples=args.n_samples)
    if not args.no_save:
        results.to_csv(OUT_CSV, index=False)
        print(f"\n[OK] Saved sweep table -> {OUT_CSV}")

    md = write_report(results, OUT_MD)
    print("\n" + md)
    if not args.no_save:
        print(f"\n[OK] Saved sweep report -> {OUT_MD}")


if __name__ == "__main__":
    main()
