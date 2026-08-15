#!/usr/bin/env python3
"""
REAL WALK-FORWARD SIMULATION — La Liga seasons with point-in-time knowledge.

Scenario
--------
At the start of a season you have ONLY the fixture list (date, teams) and the
pre-match bookmaker odds. You do NOT know any results. Starting bankroll:
$1,000,000. The simulator walks the season match-by-match in chronological
order:

  * at each match it uses ONLY information known before kick-off:
    - Elo ratings and form built from every match played BEFORE that date
      (trained on the seasons before the one being replayed, then updated
      online),
    - the real pre-match odds on the fixture sheet;
  * it computes P(home), P(draw), P(away) with the PoissonElo model, bets the
    best edge (edge = p * odds - 1) with quarter-Kelly capped at 5% of the
    bankroll and a 15% daily exposure cap;
  * after the match the real result is revealed: bankroll updated, Elo/form
    knowledge advanced (still no future info);
  * RISK RULE: if the bankroll ever drops below 10% of the start (-90%), the
    system switches to survival mode - tiny flat stakes (0.5% of bankroll), a
    relaxed edge threshold, spread across many matches.

Bookmaker / CLV
---------------
* --book b365     (default): bet at B365 prices; CLV vs the Pinnacle line.
* --book pinnacle:           bet at Pinnacle (PSH/PSD/PSA) prices; CLV vs the
                             B365 line (how much better the sharp price is).

--multi replays ALL five La Liga seasons (2021/22 .. 2025/26) with expanding-
window training, for BOTH books, and reports a multi-season distribution.

Usage:
    python demo/real_simulation.py                   # single season 22/23, b365
    python demo/real_simulation.py --book pinnacle   # sharp-priced replay
    python demo/real_simulation.py --multi           # all seasons, both books
    python demo/real_simulation.py --offline
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.poisson_elo_model import PoissonEloModel  # noqa: E402

REAL_DIR = PROJECT_ROOT / "data" / "real"
OUT_DIR = PROJECT_ROOT / "demo" / "output"

INITIAL = 1_000_000.0
ALL_SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]   # 2020/21 .. 2025/26
SEASON_LABEL = {"2021": "2020/21", "2122": "2021/22", "2223": "2022/23",
                "2324": "2023/24", "2425": "2024/25", "2526": "2025/26"}
SIM_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
EDGE_THRESHOLD = 0.03
PROB_FLOOR = 0.40
KELLY_FRACTION = 0.25
STAKE_CAP = 0.05
DAILY_CAP = 0.15
SURVIVAL_FLOOR = 0.10
SURVIVAL_EDGE = 0.01
SURVIVAL_STAKE = 0.005

RESULT_KEY = {"H": "home_win", "D": "draw", "A": "away_win"}


def download_season(season: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
        "PSH": "pin_home", "PSD": "pin_draw", "PSA": "pin_away"})
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    keep = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "result", "odds_home", "odds_draw", "odds_away",
            "pin_home", "pin_draw", "pin_away"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["date", "home_goals", "away_goals", "result"])
    return df.sort_values("date").reset_index(drop=True)


def get_season(season: str, offline: bool) -> pd.DataFrame:
    cache = REAL_DIR / f"SP1_{season}.csv"
    if offline:
        if not cache.exists():
            sys.exit(f"[FAIL] --offline but {cache} missing. Run once without --offline.")
    else:
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            cache.write_bytes(pd.read_csv(
                f"https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"
            ).to_csv(index=False).encode())
    return download_season(season)


def replay_season(train: pd.DataFrame, season: pd.DataFrame, start: float,
                  book: str = "b365", label: str = "") -> tuple:
    """Walk one season point-in-time. Returns (summary dict, per-bet DataFrame)."""
    poisson = PoissonEloModel(elo_k=20.0)
    poisson.train(train)

    bankroll = start
    bets = []
    survival_triggered = None
    daily_used, last_day = 0.0, None

    for _, r in season.iterrows():
        match_date = r["date"]
        if last_day is not None and match_date != last_day:
            daily_used = 0.0
        last_day = match_date

        p = poisson.predict(r["home_team"], r["away_team"])
        if book == "pinnacle":
            odds = {"home_win": r.get("pin_home"), "draw": r.get("pin_draw"),
                    "away_win": r.get("pin_away")}
            ref = {"home_win": r.get("odds_home"), "draw": r.get("odds_draw"),
                   "away_win": r.get("odds_away")}
        else:
            odds = {"home_win": r.get("odds_home"), "draw": r.get("odds_draw"),
                    "away_win": r.get("odds_away")}
            ref = {"home_win": r.get("pin_home"), "draw": r.get("pin_draw"),
                   "away_win": r.get("pin_away")}
        edges = {k: (p[k] * odds[k] - 1.0) for k in odds
                 if pd.notna(odds[k]) and odds[k] > 1.0}

        placed = None
        if edges:
            best = max(edges, key=edges.get)
            mode = "survival" if bankroll < SURVIVAL_FLOOR * start else "aggressive"
            threshold = SURVIVAL_EDGE if mode == "survival" else EDGE_THRESHOLD
            prob_ok = (mode == "survival") or (p[best] >= PROB_FLOOR)
            if edges[best] > threshold and prob_ok:
                odds_bet = odds[best]
                if mode == "survival":
                    stake = SURVIVAL_STAKE * bankroll
                else:
                    f = max(0.0, (p[best] * odds_bet - 1.0) / (odds_bet - 1.0))
                    stake = min(KELLY_FRACTION * f * bankroll, STAKE_CAP * bankroll)
                remaining_daily = max(0.0, DAILY_CAP * bankroll - daily_used)
                stake = min(stake, remaining_daily)
                if stake > 1.0:
                    daily_used += stake
                    placed = (best, stake, odds_bet, p[best], edges[best], mode)

        won = placed is not None and RESULT_KEY[r["result"]] == placed[0]
        if placed is not None:
            outcome, stake, odds_bet, prob, edge, mode = placed
            profit = (stake * (odds_bet - 1.0)) if won else (-stake)
            bankroll += profit
            line = ref[outcome]
            # CLV = bet price / reference line - 1 (positive = beat the line)
            clv = (odds_bet / line - 1.0) * 100 if pd.notna(line) and line > 1.0 else np.nan
            bets.append({
                "date": match_date, "match": f"{r['home_team']} vs {r['away_team']}",
                "outcome": outcome, "prob": prob, "odds": odds_bet, "edge_pct": edge * 100,
                "stake": stake, "mode": mode, "result": r["result"], "win": int(won),
                "profit": profit, "clv_pct": clv, "bankroll": bankroll,
            })
            if mode == "survival" and survival_triggered is None:
                survival_triggered = match_date

        poisson._update_elo(r["home_team"], r["away_team"],
                            int(r["home_goals"]), int(r["away_goals"]))

    bets_df = pd.DataFrame(bets)
    n = len(bets_df)
    wins = int(bets_df["win"].sum()) if n else 0
    peak, max_dd = start, 0.0
    for b in bets_df["bankroll"] if n else []:
        peak = max(peak, b)
        max_dd = max(max_dd, (peak - b) / peak)
    clv_ok = bets_df["clv_pct"].dropna() if n else pd.Series(dtype=float)
    clv_t = clv_p = None
    if len(clv_ok) >= 2:
        from scipy import stats
        clv_t, clv_p = stats.ttest_1samp(clv_ok, 0.0)
    summary = {
        "book": book, "season": label,
        "final": bankroll, "roi": (bankroll / start - 1) * 100,
        "n_bets": n, "wins": wins,
        "strike": wins / n if n else float("nan"),
        "avg_odds": bets_df["odds"].mean() if n else float("nan"),
        "avg_edge": bets_df["edge_pct"].mean() if n else float("nan"),
        "max_dd": max_dd,
        "avg_clv": float(clv_ok.mean()) if len(clv_ok) else float("nan"),
        "clv_t": float(clv_t) if clv_t is not None else float("nan"),
        "clv_p": float(clv_p) if clv_p is not None else float("nan"),
        "survival": survival_triggered.date() if survival_triggered else None,
    }
    return summary, bets_df


def _print_summary(r: dict):
    print(f"    {r['book']:<9} {r['season']:<10} final ${r['final']:>12,.0f} "
          f"ROI {r['roi']:+7.1f}%  bets {r['n_bets']:>3}  strike {r['strike']:.0%}  "
          f"maxDD {r['max_dd']:.0%}  CLV {r['avg_clv']:+.2f}%")


def run_multi(offline: bool, start: float):
    print("=" * 92)
    print("MULTI-SEASON REAL REPLAY — La Liga 2021/22 .. 2025/26, both books")
    print("Every season: expanding-window train on all previous seasons, then a")
    print("point-in-time walk. The five ROIs form a small distribution.")
    print("=" * 92)
    rows = []
    for i, sim_season in enumerate(SIM_SEASONS):
        train = pd.concat([get_season(s, offline) for s in ALL_SEASONS[:i + 1]],
                          ignore_index=True).sort_values("date").reset_index(drop=True)
        season = get_season(sim_season, offline)
        print(f"\n  --- {SEASON_LABEL[sim_season]} (trained on {len(train)} prior matches) ---")
        for book in ["b365", "pinnacle"]:
            summary, bets_df = replay_season(train, season, start, book=book,
                                             label=SEASON_LABEL[sim_season])
            rows.append(summary)
            _print_summary(summary)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "real_simulation_multi.csv", index=False)

    print("\n" + "=" * 92)
    print("DISTRIBUTION ACROSS FIVE SEASONS (the real answer to 'is it luck?')")
    print("=" * 92)
    from scipy import stats as _st
    for book in ["b365", "pinnacle"]:
        sub = res[res["book"] == book]
        rois = sub["roi"].to_numpy()
        mean, sem = rois.mean(), _st.sem(rois)
        ci = (mean - 1.96 * sem, mean + 1.96 * sem)
        clvs = sub["avg_clv"].dropna()
        line = f"  {book} book: ROIs {['%+.1f%%' % v for v in rois]} | "
        line += f"mean {mean:+.1f}% (95% CI {ci[0]:+.1f}..{ci[1]:+.1f}%) | "
        line += f"median {np.median(rois):+.1f}% | positive {int((rois > 0).sum())}/{len(rois)}"
        if len(clvs):
            t, p = _st.ttest_1samp(clvs, 0.0)
            line += f" | avg CLV {clvs.mean():+.2f}% (t={t:.1f}, p={p:.3f})"
        print(line)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        for book, color in [("b365", "#2E86AB"), ("pinnacle", "#06D6A0")]:
            sub = res[res["book"] == book]
            ax.plot(range(len(sub)), sub["roi"], marker="o", label=f"{book} book", color=color)
        ax.axhline(0, color="#9aa0a6", ls="--", lw=1)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["season"], rotation=20, ha="right")
        ax.set_ylabel("Season ROI (%)")
        ax.set_title("Real walk-forward replay — ROI by season and book")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "real_simulation_multi.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  [OK] Saved {OUT_DIR / 'real_simulation_multi.csv'} / .png")
    except Exception as e:
        print(f"  [warn] plot failed: {e}")

    doc = PROJECT_ROOT / "docs" / "09_real_walkforward_simulation.md"
    lines = [
        "# Real Walk-Forward Simulation — multi-season (La Liga 2021/22 .. 2025/26)",
        "",
        "Point-in-time replay of five real La Liga seasons with a $1M start. Each",
        "season trains on all seasons before it (expanding window) and then walks",
        "match-by-match using ONLY information known before kick-off (online Elo/",
        "form, real pre-match odds). Staking: quarter-Kelly capped at 5% of",
        "bankroll, 15% daily cap, survival mode after a -90% drawdown.",
        "",
        "```",
        res.to_string(index=False),
        "```",
        "",
        "## The multi-season verdict (small sample, be honest)",
        "",
        "- Five seasons is still a small sample: the 95% CI on mean ROI is wide.",
        "- CLV vs the sharp line (Pinnacle) is the most reliable real-world signal",
        "  of whether the model's prices beat the market.",
        "- Compare books: betting at B365 vs Pinnacle prices shows how much the",
        "  soft bookmaker margin costs.",
        "",
        "*(Saved by `demo/real_simulation.py --multi`; per-season log in",
        "`demo/output/real_simulation_multi.csv`.)*",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [OK] Wrote {doc}")


def run_single(season_code: str, book: str, offline: bool, start: float):
    train_codes = [s for s in ALL_SEASONS if s < season_code]
    train = pd.concat([get_season(s, offline) for s in train_codes],
                      ignore_index=True).sort_values("date").reset_index(drop=True)
    season = get_season(season_code, offline)
    print(f"  Training knowledge : {train_codes} ({len(train)} matches, "
          f"latest {train['date'].max().date()})")
    print(f"  Simulated league   : {SEASON_LABEL[season_code]} ({len(season)} matches, "
          f"{season['date'].min().date()} -> {season['date'].max().date()})")
    print(f"  Book               : {book}")
    summary, bets_df = replay_season(train, season, start, book=book,
                                     label=SEASON_LABEL[season_code])

    print("\n" + "-" * 84)
    print("FINAL RESULT (end of league)")
    print("-" * 84)
    _print_summary(summary)
    print(f"  Survival mode from : {summary['survival'] or 'never'}")
    if summary["n_bets"] >= 2:
        print(f"  Real CLV vs reference line: {summary['avg_clv']:+.2f}% "
              f"(t={summary['clv_t']:.2f}, p={summary['clv_p']:.3f})")

    # save artifacts
    bets_df.to_csv(OUT_DIR / f"real_simulation_{season_code}.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        dates = pd.to_datetime(bets_df["date"])
        ax.plot(dates, bets_df["bankroll"], color="#2E86AB", lw=1.6)
        ax.axhline(INITIAL, color="#9aa0a6", ls="--", lw=1, label="start $1M")
        ax.axhline(SURVIVAL_FLOOR * INITIAL, color="#d1495b", ls=":", lw=1.2,
                   label="survival floor")
        if len(bets_df):
            ax.scatter(dates, bets_df["bankroll"], s=8, color="#F18F01",
                       label="bets", alpha=0.6)
        ax.set_title(f"Real walk-forward — La Liga {SEASON_LABEL[season_code]} ({book} book)")
        ax.set_ylabel("Bankroll ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"real_simulation_{season_code}_{book}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Saved demo/output/real_simulation_{season_code}.csv "
              f"+ real_simulation_{season_code}_{book}.png")
    except Exception as e:
        print(f"  [warn] plot failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Real walk-forward La Liga simulation")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--start", type=float, default=INITIAL)
    parser.add_argument("--multi", action="store_true", help="replay all 5 seasons, both books")
    parser.add_argument("--book", choices=["b365", "pinnacle"], default="b365")
    parser.add_argument("--season", default="2223", help="single-season replay (default 2223)")
    args = parser.parse_args()

    if args.multi:
        run_multi(args.offline, args.start)
        return

    print("=" * 84)
    print(f"REAL WALK-FORWARD SIMULATION — La Liga {SEASON_LABEL[args.season]}, $1M start")
    print("You only know the fixture list + pre-match odds; results are revealed")
    print("chronologically. Point-in-time Elo/form knowledge. Real CLV.")
    print("=" * 84)
    run_single(args.season, args.book, args.offline, args.start)


if __name__ == "__main__":
    main()
