#!/usr/bin/env python3
"""
HIDDEN SIGNALS ON REAL DATA — multi-bookmaker consensus, dispersion and the
public-vs-sharp split, tested on real Serie A matches.

The "hidden signals" thesis, tested directly
--------------------------------------------
1. CONSENSUS & DISPERSION — the cached football-data CSVs carry up to a dozen
   bookmakers (B365, Bwin, Betfair, Pinnacle, Max/Avg ...).  Their AGREEMENT
   (consensus implied probability) is itself a signal, and their DISAGREEMENT
   (dispersion) is the market telling you a match is hard to price.

2. PUBLIC vs SHARP split — B365 (soft public) vs Pinnacle (sharp).  When the
   sharp line disagrees with the public line, which way does value lie?

3. CLV — betting at the public price while using the sharp line as a signal:
   does that beat the closing market?  (CLV = taken price / sharp line - 1.)

4. The DynamicThinkingLayer on real Serie A 25/26 with all signals fused.

Protocol: train on real Serie A 2020/21-2023/24, walk 2025/26 point-in-time
(no future info).  Results -> backtests/results/hidden_signals_results.csv
+ docs/13_hidden_signals.md

Usage:
    python scripts/12_hidden_signals.py --offline
"""

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.real_data import REAL_DIR, SEASON_LABEL  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = ["2021", "2122", "2223", "2324"]
TEST_SEASON = "2526"

# bookmaker triples available in the football-data.co.uk CSVs
BOOKS = {
    "B365": ("B365H", "B365D", "B365A"),
    "Bwin": ("BWH", "BWD", "BWA"),
    "Betfair": ("BFDH", "BFDD", "BFDA"),
    "Betvictor": ("BVH", "BVD", "BVA"),
    "Bet365": ("BMGMH", "BMGMD", "BMGMA"),
    "Pinnacle": ("PSH", "PSD", "PSA"),
}
OUTCOMES = ("home_win", "draw", "away_win")
# NOTE: this script builds probability vectors ordered [home, draw, away].
# The shared CLASS_MAP ({"A": 0, "D": 1, "H": 2}) indexes [away, draw, home]
# and would silently swap home/away in any argmax comparison.  Use an explicit
# [home, draw, away] result index here instead.
RESULT_INDEX = {"H": 0, "D": 1, "A": 2}


def load_serie_a(season: str) -> pd.DataFrame:
    df = pd.read_csv(REAL_DIR / f"I1_{season}.csv")
    df = df.rename(columns={"Date": "date", "HomeTeam": "home_team",
                            "AwayTeam": "away_team", "FTHG": "home_goals",
                            "FTAG": "away_goals", "FTR": "result"})
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "result"])
    return df.sort_values("date").reset_index(drop=True)


def book_odds(row, cols) -> dict:
    if all(pd.notna(row[c]) and row[c] > 1 for c in cols):
        return {"home_win": float(row[cols[0]]), "draw": float(row[cols[1]]),
                "away_win": float(row[cols[2]])}
    return None


def market_signals(row) -> dict:
    """Consensus + dispersion + public/sharp split from all available books."""
    mats = []
    for name, cols in BOOKS.items():
        o = book_odds(row, cols)
        if o is not None:
            p = np.array([1 / o[k] for k in OUTCOMES], dtype=float)
            mats.append(p / p.sum())
    if not mats:
        return None
    mat = np.array(mats)
    consensus = mat.mean(axis=0)
    dispersion = float(mat.std(axis=0).mean())
    public = book_odds(row, BOOKS["B365"])
    sharp = book_odds(row, BOOKS["Pinnacle"])
    split = None
    if public and sharp:
        pp = np.array([1 / public[k] for k in OUTCOMES]); pp /= pp.sum()
        sp = np.array([1 / sharp[k] for k in OUTCOMES]); sp /= sp.sum()
        split = sp - pp
    return {"n_books": len(mats), "consensus": consensus, "dispersion": dispersion,
            "split": split, "public": public, "sharp": sharp}


def exp1_dispersion(test: pd.DataFrame) -> dict:
    """Do high-dispersion matches resist prediction? (signal validity test)"""
    rows = []
    for _, r in test.iterrows():
        sig = market_signals(r)
        if sig is None:
            continue
        y = RESULT_INDEX[r["result"]]
        rows.append({"dispersion": sig["dispersion"],
                     "correct_by_consensus": int(np.argmax(sig["consensus"]) == y),
                     "y": y})
    df = pd.DataFrame(rows)
    lo = df[df["dispersion"] <= df["dispersion"].median()]
    hi = df[df["dispersion"] > df["dispersion"].median()]
    return {
        "n": len(df),
        "consensus_acc_low_disp": round(float(lo["correct_by_consensus"].mean()), 4),
        "consensus_acc_high_disp": round(float(hi["correct_by_consensus"].mean()), 4),
        "mean_dispersion": round(float(df["dispersion"].mean()), 4),
    }


def exp2_split_clv(test: pd.DataFrame) -> dict:
    """Bet at the public (B365) price when the sharp split is large; CLV."""
    clvs, bets = [], []
    for _, r in test.iterrows():
        sig = market_signals(r)
        if sig is None or sig["split"] is None:
            continue
        split = sig["split"]
        # bet the outcome the sharp line is most bullish on vs public
        best = int(np.argmax(split))
        public_o = [sig["public"][k] for k in OUTCOMES][best]
        sharp_o = [sig["sharp"][k] for k in OUTCOMES][best]
        clv = (public_o / sharp_o - 1) * 100
        clvs.append(clv)
        bets.append(RESULT_INDEX[r["result"]] == best)
    clvs = np.array(clvs)
    from scipy import stats
    t, p = stats.ttest_1samp(clvs, 0.0) if len(clvs) >= 2 else (np.nan, np.nan)
    return {
        "n": len(clvs),
        "avg_clv_pct": round(float(clvs.mean()), 2) if len(clvs) else float("nan"),
        "clv_t": round(float(t), 2) if not np.isnan(t) else float("nan"),
        "clv_p": round(float(p), 4) if not np.isnan(p) else float("nan"),
        "strike_at_public": round(float(np.mean(bets)), 4) if bets else float("nan"),
        "positive": int((clvs > 0).sum()) if len(clvs) else 0,
    }


def exp3_dynamic_layer(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """DynamicThinkingLayer on real Serie A 25/26 with real market signals."""
    from models.dynamic_thinking import DynamicThinkingLayer
    layer = DynamicThinkingLayer(train_df=train, bankroll=1_000_000.0)
    for day, (_, r) in enumerate(test.iterrows()):
        sig = market_signals(r)
        if sig is None or sig["public"] is None or sig["sharp"] is None:
            continue
        extra = []
        for name, cols in BOOKS.items():
            if name in ("B365", "Pinnacle"):
                continue
            o = book_odds(r, cols)
            if o is not None:
                extra.append(o)
        decision = layer.think(r["home_team"], r["away_team"],
                               sig["public"], sig["sharp"],
                               extra_books=extra, current_day=day)
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        layer.observe(r["home_team"], r["away_team"], hg, ag, r["result"],
                      decision, sig["public"], current_day=day)
    s = layer.summary()
    s["base_refits"] = getattr(layer.base, "refits", 0)
    return s


def main():
    parser = argparse.ArgumentParser(description="Hidden-signals real-data test")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("HIDDEN SIGNALS ON REAL DATA — consensus, dispersion, public-vs-sharp")
    print("=" * 78)

    train = pd.concat([load_serie_a(s) for s in TRAIN_SEASONS],
                      ignore_index=True).sort_values("date").reset_index(drop=True)
    test = load_serie_a(TEST_SEASON)
    print(f"  Train: {len(train)} real Serie A matches (2020/21-2023/24)")
    print(f"  Test : {len(test)} real Serie A matches (2025/26)\n")

    results = []

    # Exp 1: does dispersion (market disagreement) signal unpredictability?
    e1 = exp1_dispersion(test)
    print("  [1/3] Dispersion vs predictability (consensus accuracy)")
    print(f"        low-dispersion matches: {e1['consensus_acc_low_disp']:.1%} correct")
    print(f"        high-dispersion matches: {e1['consensus_acc_high_disp']:.1%} correct")
    results.append({"experiment": "dispersion-vs-predictability", "method": "consensus",
                    **e1})

    # Exp 2: sharp-vs-public split -> bet at public price, measure CLV
    e2 = exp2_split_clv(test)
    print(f"\n  [2/3] Sharp-vs-public split (bet at public price, CLV vs sharp)")
    print(f"        n={e2['n']}  avg CLV {e2['avg_clv_pct']:+.2f}%  "
          f"(t={e2['clv_t']}, p={e2['clv_p']})  positive {e2['positive']}/{e2['n']}")
    results.append({"experiment": "sharp-public-split", "method": "CLV", **e2})

    # Exp 3: the dynamic thinking layer on real matches
    print("\n  [3/3] DynamicThinkingLayer on real Serie A 25/26 (multi-book signals)")
    e3 = exp3_dynamic_layer(train, test)
    print(f"        final ${e3['final_bankroll']:,.0f}  ROI {e3['roi_pct']:+.1f}%  "
          f"bets {e3['n_bets']}  strike {e3['strike_rate']:.0f}%  "
          f"refits {e3['base_refits']}  survival {e3['survival']}")
    results.append({"experiment": "dynamic-layer-real-2526", "method": "DynamicThinking",
                    **e3})

    res = pd.DataFrame(results)
    res.to_csv(RESULTS_DIR / "hidden_signals_results.csv", index=False)
    print(f"\n[OK] Saved {RESULTS_DIR / 'hidden_signals_results.csv'}")

    _write_doc(res, e1, e2, e3)


def _write_doc(res, e1, e2, e3):
    lines = [
        "# Hidden Signals on Real Data",
        "",
        "The cached football-data.co.uk CSVs carry up to a dozen bookmakers per",
        "match (B365, Bwin, Betfair, Betvictor, Bet365, Pinnacle, Max/Avg).  This",
        "experiment tests the 'hidden signals' thesis on real Serie A 2025/26",
        "matches with point-in-time knowledge only:",
        "",
        "1. **Consensus & dispersion** — do matches where the market disagrees",
        "   resist prediction?",
        "2. **Public vs sharp split** — betting the public (B365) price when the",
        "   sharp (Pinnacle) line disagrees: does it beat the closing market (CLV)?",
        "3. **The DynamicThinkingLayer** — the fully self-refitting model fused",
        "   with all real signals, walked over 2025/26.",
        "",
        "## Dispersion vs predictability",
        "",
        f"Consensus accuracy on low-dispersion matches: **{e1['consensus_acc_low_disp']:.1%}**",
        f"Consensus accuracy on high-dispersion matches: **{e1['consensus_acc_high_disp']:.1%}**",
        "",
        "## Sharp-vs-public split CLV",
        "",
        f"n={e2['n']} · avg CLV **{e2['avg_clv_pct']:+.2f}%** "
        f"(t={e2['clv_t']}, p={e2['clv_p']}) · positive {e2['positive']}/{e2['n']}",
        "",
        "## Dynamic layer on real 2025/26",
        "",
        f"Final **${e3['final_bankroll']:,.0f}** · ROI **{e3['roi_pct']:+.1f}%** · "
        f"bets {e3['n_bets']} · strike {e3['strike_rate']}% · "
        f"online refits {e3['base_refits']}",
        "",
        "*(Saved by `scripts/12_hidden_signals.py`; full numbers in",
        "`backtests/results/hidden_signals_results.csv`.)*",
    ]
    doc = PROJECT_ROOT / "docs" / "13_hidden_signals.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {doc}")


if __name__ == "__main__":
    main()
