#!/usr/bin/env python3
"""
14 — Different sport: ATP tennis walk-forward (strict no-future-knowledge).

Football agent_sim is 3-outcome (home/draw/away).  Tennis is a genuinely
DIFFERENT sport: 2-outcome matches, no draws, a surface structure, and a
different data schema (winner/loser + ATP ranks instead of goals).  This
script re-applies the same methodology — an ADAPTIVE online model that only
sees matches before the current moment, a strict chronological walk with a
leakage audit, edge-based staking, survival mode, and baselines — on real
ATP data with real odds.

Data: real ATP 2016-2025 from tennis-data.co.uk (free, no key).  B365 = the
public price you can actually bet; Pinnacle = the sharp reference (CLV).

    python scripts/14_tennis_walkforward.py            # walks 2024 + 2025
    python scripts/14_tennis_walkforward.py --walk 2025
    python scripts/14_tennis_walkforward.py --offline  # use cached CSVs

Outputs (backtests/results/tennis_walkforward/):
    tennis_summary.csv               aggregate per model + per walk year
    tennis_bets_<year>.csv           full betting transaction ledger
    tennis_opportunities_<year>.csv  every match evaluated + audit fields
    tennis_by_surface_<year>.csv     profit/ROI/accuracy by court surface
    tennis_equity_<year>.csv         bankroll over time
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_sim.tennis import fetch_seasons          # noqa: E402

RESULTS = ROOT / "backtests" / "results" / "tennis_walkforward"
TRAIN_YEARS = list(range(2016, 2024))               # 2016-2023 train
WALK_YEARS = [2024, 2025]                           # untouched walk periods

EDGE_BASE = 0.04            # tennis is a lower-margin market than football
PROB_FLOOR = 0.55           # 2-outcome: need a clear favourite
MIN_ODDS = 1.35
FLAT_STAKE = 10_000.0
KELLY_FRACTION = 0.25
STAKE_CAP_FRAC = 0.02
DAILY_CAP_FRAC = 0.15
SURVIVAL_FLOOR = 0.10
SURVIVAL_EDGE = 0.015
SURVIVAL_PROB_FLOOR = 0.45
SURVIVAL_STAKE_FRAC = 0.005

ELO_START = 1500.0
ELO_K = 32.0
ELO_SCALE = 400.0
SURFACE_WEIGHT = 0.55        # blend of surface-specific and overall rating


class TennisElo:
    """Online 2-outcome Elo that learns ONLY from matches already played.

    Keeps one overall rating per player plus a per-surface rating; the match
    probability uses a weighted blend.  This is the tennis analogue of the
    football PoissonElo — fully adaptive, zero future knowledge.
    """

    def __init__(self, seed: int = 42):
        self.overall: dict[str, float] = {}
        self.by_surface: dict[str, dict[str, float]] = defaultdict(dict)
        self.rng = np.random.default_rng(seed)

    def prob(self, player_a: str, player_b: str, surface: str) -> float:
        """P(player_a beats player_b) on this surface, from ratings ONLY
        built on matches that already happened."""
        ra = self.overall.get(player_a, ELO_START)
        rb = self.overall.get(player_b, ELO_START)
        sa = self.by_surface[surface].get(player_a, ELO_START)
        sb = self.by_surface[surface].get(player_b, ELO_START)
        ra = SURFACE_WEIGHT * sa + (1 - SURFACE_WEIGHT) * ra
        rb = SURFACE_WEIGHT * sb + (1 - SURFACE_WEIGHT) * rb
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / ELO_SCALE))

    def update(self, winner: str, loser: str, surface: str,
               wsets: int, lsets: int):
        """Standard Elo update; margin (sets difference) scales K modestly."""
        for table, rw, rl in (
            (self.overall, self.overall.get(winner, ELO_START),
             self.overall.get(loser, ELO_START)),
            (self.by_surface[surface],
             self.by_surface[surface].get(winner, ELO_START),
             self.by_surface[surface].get(loser, ELO_START)),
        ):
            exp_w = 1.0 / (1.0 + 10.0 ** ((rl - rw) / ELO_SCALE))
            margin = max(0, int(wsets or 2) - int(lsets or 0))
            k = ELO_K * (1.0 + 0.25 * margin)
            table[winner] = rw + k * (1.0 - exp_w)
            table[loser] = rl + k * (0.0 - (1.0 - exp_w))


def implied_probs(odds_w: float, odds_l: float) -> dict:
    if not (odds_w > 1.0 and odds_l > 1.0):
        return {}
    iw, il = 1.0 / odds_w, 1.0 / odds_l
    tot = iw + il
    return {"winner": iw / tot, "loser": il / tot}


def run_walk(year: int, mode: str, seed: int,
             bankroll: float, results: Path) -> dict:
    """Walk one full season chronologically for one policy."""
    train = fetch_seasons(TRAIN_YEARS)
    walk = _load_walk(year)

    # ---- initialise the model on training data ONLY (before the walk) ----
    elo = TennisElo(seed=seed)
    for _, m in train.iterrows():
        elo.update(m["winner"], m["loser"], m["surface"], 2, 0)
    last_known = train["date"].max()

    start = float(bankroll)
    bank = start
    peak = start
    n_bets = n_wins = 0
    total_staked = 0.0
    daily_used = 0.0
    last_day = None
    survival = False
    survival_day = None
    n_leaks = 0
    rows = []
    rng = np.random.default_rng(seed)
    acc_better = acc_lower = 0     # model prob >= 0.5 vs actual winner
    brier_sum = 0.0
    clv_list = []

    # ---- two-phase daily walk (strict no-future-knowledge) --------------
    # The football engine resolves a day's matches before offering new ones.
    # The previous tennis walk settled + learned per row, so a player in two
    # matches on the SAME calendar date could leak the first result into the
    # second prediction's Elo.  Now every match of a day is predicted and
    # decided against the Elo state as of the PREVIOUS day; settlement and
    # Elo updates happen only after all of the day's decisions are locked,
    # so the leak audit (data_cutoff > day) is genuinely airtight.
    for day, grp in walk.groupby("date", sort=True):
        if last_day is not None and day != last_day:
            daily_used = 0.0
        last_day = day
        pending = []   # (match, decision, stake, odds, pins, leak, row)

        # ---- phase 1: predict + decide every match of the day ------------
        for _, m in grp.iterrows():
            winner, loser = m["winner"], m["loser"]
            surf = m["surface"]
            odds_w = float(m.get("odds_winner", np.nan))
            odds_l = float(m.get("odds_loser", np.nan))
            pin_w = float(m.get("pin_winner", np.nan))
            pin_l = float(m.get("pin_loser", np.nan))

            # the ONLY audit: features must come from before this match
            data_cutoff = last_known
            leak = bool(pd.notna(data_cutoff) and data_cutoff > day)

            # model probs built ONLY from matches before this day
            p_w = elo.prob(winner, loser, surf)
            p_l = 1.0 - p_w
            probs = {"winner": float(p_w), "loser": float(p_l)}

            # pick accuracy + Brier on the favourite outcome (whole universe)
            acc_better += int(p_w >= 0.5)        # model favoured the real winner
            acc_lower += 1
            brier_sum += (p_w - 1.0) ** 2        # target: winner always = 1

            decision = None
            reason = "no edge"
            stake = 0.0
            edge = 0.0
            conf = float(np.clip((max(probs.values()) - 0.5) / 0.5, 0.0, 1.0))

            if not leak:
                if mode == "ml":
                    best = "winner" if p_w >= p_l else "loser"
                    best_odds = odds_w if best == "winner" else odds_l
                    if pd.notna(best_odds) and best_odds > 1.0:
                        edge = probs[best] * best_odds - 1.0
                        floor = SURVIVAL_PROB_FLOOR if survival else PROB_FLOOR
                        thresh = SURVIVAL_EDGE if survival else EDGE_BASE
                        if edge > thresh and probs[best] >= floor \
                                and best_odds >= MIN_ODDS:
                            if survival:
                                stake = SURVIVAL_STAKE_FRAC * bank
                            else:
                                kelly = max(0.0, edge / (best_odds - 1.0))
                                stake = min(KELLY_FRACTION * kelly * bank,
                                            STAKE_CAP_FRAC * bank)
                            stake = min(stake, max(
                                0.0, DAILY_CAP_FRAC * bank - daily_used))
                            if stake >= 1.0:
                                decision = best
                                reason = (f"edge {edge:+.1%} > {thresh:+.1%}"
                                          + (", survival" if survival else ""))
                            else:
                                reason = "stake below minimum (daily cap)"
                        elif edge <= 0:
                            reason = "no positive edge"
                        else:
                            reason = (f"prob {probs[best]:.0%} < {floor:.0%}"
                                      if probs[best] < floor
                                      else f"odds {best_odds:.2f} < {MIN_ODDS}")
                elif mode == "implied":
                    ip = implied_probs(odds_w, odds_l)
                    if ip and max(ip.values()) >= 0.55:
                        best = max(ip, key=ip.get)
                        decision, stake = best, FLAT_STAKE
                        reason = f"implied {ip[best]:.0%} >= 55%"
                elif mode == "random" and pd.notna(odds_w) and odds_w > 1.0:
                    decision = str(rng.choice(["winner", "loser"]))
                    stake = FLAT_STAKE
                    reason = "random"
                elif mode == "nobet":
                    reason = "no-bet baseline"

            bank_before = round(bank, 2)
            if leak:
                n_leaks += 1
                decision = None
                reason = "DATA LEAKAGE — prediction invalidated"
                stake = 0.0

            rows.append({
                "date": day, "tournament": m.get("tournament", ""),
                "round": m.get("round", ""), "surface": surf,
                "match": f"{winner} vs {loser}",
                "wrank": m.get("wrank", np.nan), "lrank": m.get("lrank", np.nan),
                "prob_winner": round(p_w, 4),
                "odds_winner": odds_w, "odds_loser": odds_l,
                "pin_winner": pin_w, "pin_loser": pin_l,
                "edge": round(edge, 4), "confidence": round(conf, 3),
                "decision": decision, "reason": reason, "stake": round(stake, 2),
                "result": "",
                "leak_flag": int(leak), "invalidated": int(leak),
                "data_cutoff": str(data_cutoff.date()) if pd.notna(data_cutoff) else "",
                "bankroll_before": bank_before, "profit": 0.0,
                "bankroll_after": round(bank, 2),
            })
            pending.append((m, decision, stake, odds_w, odds_l,
                            pin_w, pin_l, leak, rows[-1]))

        # ---- phase 2: settle + learn after ALL of the day's decisions ----
        for m, decision, stake, odds_w, odds_l, pin_w, pin_l, leak, row in pending:
            winner, loser = m["winner"], m["loser"]
            surf = m["surface"]
            won = False
            profit = 0.0
            if decision is not None and not leak:
                won = (decision == "winner")
                odds_taken = odds_w if decision == "winner" else odds_l
                profit = stake * (odds_taken - 1.0) if won else -stake
                bank += profit
                peak = max(peak, bank)
                daily_used += stake
                n_bets += 1
                n_wins += int(won)
                total_staked += stake
                if bank < SURVIVAL_FLOOR * start and not survival:
                    survival, survival_day = True, day
                # CLV vs Pinnacle (sharp) — positive means we beat the line
                sharp = pin_w if decision == "winner" else pin_l
                if pd.notna(sharp) and sharp > 1.0 and odds_taken > 1.0:
                    clv_list.append((sharp - odds_taken) / odds_taken)

            row["result"] = "winner" if won else ("loser" if decision else "")
            row["profit"] = round(profit, 2)
            row["bankroll_after"] = round(bank, 2)

            # learn the result AFTER it happened (adaptive, chronological)
            elo.update(winner, loser, surf,
                       int(m.get("wsets", 2)), int(m.get("lsets", 0)))
            last_known = day

    df = pd.DataFrame(rows)
    bets = df[(df["decision"].notna()) & (df["invalidated"] == 0)]
    profit_sum = float(bets["profit"].sum())
    staked = float(bets["stake"].sum())
    roi = profit_sum / staked * 100 if staked else 0.0
    accuracy = acc_better / acc_lower if acc_lower else 0.0
    brier = brier_sum / acc_lower if acc_lower else 0.0
    mean_clv = float(np.mean(clv_list)) if clv_list else 0.0

    # max drawdown from bankroll path
    peak2, max_dd = -np.inf, 0.0
    for b in bets["bankroll_after"]:
        peak2 = max(peak2, b)
        max_dd = max(max_dd, (peak2 - b) / peak2 if peak2 else 0.0)

    # by-surface
    surf_rows = []
    for s, sub in bets.groupby("surface"):
        sp = float(sub["profit"].sum())
        ss = float(sub["stake"].sum())
        surf_rows.append({"surface": s, "n_bets": len(sub),
                          "wins": int((sub["profit"] > 0).sum()),
                          "profit": round(sp, 2),
                          "roi_pct": round(sp / ss * 100, 2) if ss else 0.0})
    pd.DataFrame(surf_rows).to_csv(
        results / f"tennis_by_surface_{year}.csv", index=False)

    # equity path
    pd.DataFrame(bets[["date", "bankroll_after"]]) \
        .to_csv(results / f"tennis_equity_{year}.csv", index=False)

    df.to_csv(results / f"tennis_opportunities_{year}.csv", index=False)
    bets.to_csv(results / f"tennis_bets_{year}.csv", index=False)

    return {"walk_year": year, "mode": mode, "seed": seed,
            "matches": int(acc_lower), "bets": int(n_bets),
            "wins": int(n_wins),
            "win_rate": round(n_wins / n_bets, 4) if n_bets else 0.0,
            "accuracy": round(accuracy, 4), "brier": round(brier, 4),
            "start_bankroll": start, "final_bankroll": round(bank, 2),
            "total_profit": round(profit_sum, 2),
            "total_staked": round(staked, 2), "roi_pct": round(roi, 2),
            "mean_clv_pct": round(mean_clv * 100, 3),
            "max_drawdown": round(max_dd, 4), "n_leak_flags": n_leaks,
            "survival": survival, "survival_day": str(survival_day or "")}


def _load_walk(year):
    from agent_sim.tennis import fetch_season
    df = fetch_season(year)
    # keep the wsets/lsets columns used for margin-weighted Elo updates
    xlsx = ROOT / "data" / "tennis" / f"atp_{year}.xlsx"
    if xlsx.exists():
        raw = pd.read_excel(xlsx)
        df = df.merge(raw[["Date", "Winner", "Loser", "Wsets", "Lsets"]]
                      .rename(columns={"Date": "date", "Winner": "winner",
                                       "Loser": "loser"}),
                      on=["date", "winner", "loser"], how="left")
    return df.sort_values("date").reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--walk", type=int, nargs="+", default=WALK_YEARS)
    p.add_argument("--bankroll", type=float, default=1_000_000.0)
    p.add_argument("--offline", action="store_true",
                   help="use cached data/tennis CSVs (no network)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        print("[14] caching ATP seasons:", *TRAIN_YEARS, "train +",
              *args.walk, "walk")
        for y in TRAIN_YEARS + args.walk:
            df = fetch_seasons([y])
            print(f"  ATP {y}: {len(df)} matches")

    all_rows = []
    for mode in ["ml", "implied", "random", "nobet"]:
        for y in args.walk:
            print(f"[14] walking ATP {y} with policy '{mode}'...")
            row = run_walk(y, mode, args.seed, args.bankroll, RESULTS)
            all_rows.append(row)
            print(f"      bets={row['bets']} roi={row['roi_pct']:+.1f}% "
                  f"acc={row['accuracy']:.1%} brier={row['brier']:.3f} "
                  f"clv={row['mean_clv_pct']:+.2f}% leaks={row['n_leak_flags']}")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(RESULTS / "tennis_summary.csv", index=False)
    print("-" * 64)
    print("[14] summary -> backtests/results/tennis_walkforward/"
          "tennis_summary.csv")
    print(summary[["walk_year", "mode", "bets", "win_rate", "roi_pct",
                   "mean_clv_pct", "max_drawdown"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
