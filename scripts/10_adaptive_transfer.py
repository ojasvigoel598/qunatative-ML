#!/usr/bin/env python3
"""
ADAPTIVE CROSS-LEAGUE / CROSS-SPORT TRANSFER EXPERIMENT.

Question: can ONE model be pointed at a new league (or sport) it was never
trained on, and ADAPT online instead of degrading like a frozen model?

Design (point-in-time, zero leakage)
------------------------------------
* Train the base PoissonElo + ML layer on real matches of one league
  (e.g. Serie A 2020/21-2023/24).
* Walk the target matches chronologically. For EVERY match:
      probs = model.predict(home, away)      # only past info
      record metrics
      model.observe(home, away, goals, result)   # reveal -> online update
  The ADAPTIVE model refits its ML layer on a rolling window when scheduled or
  when its rolling Brier drifts.  The STATIC control never refits (its Elo and
  form still update online), so the difference isolates the value of adapting.

Experiments
-----------
  A. WITHIN-LEAGUE unseen season : train Serie A -> test Serie A 2025/26
  B. CROSS-LEAGUE transfer      : train Serie A -> test La Liga 2025/26
  C. CROSS-LEAGUE transfer      : train Serie A -> test EPL 2025/26
  D. CROSS-SPORT                : train football -> test a synthetic
                                  high-scoring "basketball-like" league
                                  (draws nearly impossible, ~2.4 goals/team)

Results -> backtests/results/adaptive_transfer_results.csv + docs/10_adaptive_transfer.md

Usage:
    python scripts/10_adaptive_transfer.py          # download real data
    python scripts/10_adaptive_transfer.py --offline
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

from data.real_data import get_season, LEAGUES  # noqa: E402
from models.adaptive_model import AdaptiveMatchPredictor, CLASS_MAP, brier  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = ["2021", "2122", "2223", "2324"]   # 2020/21 .. 2023/24
TEST_SEASON = "2526"                                # 2025/26 (unseen)


def walk(predictor, test: pd.DataFrame) -> dict:
    """Point-in-time walk: predict, record, reveal, adapt. Returns metrics."""
    y_true, prob_rows = [], []
    for _, r in test.iterrows():
        p = predictor.predict(r["home_team"], r["away_team"])
        prob_vec = np.array([p["away_win"], p["draw"], p["home_win"]])
        y_true.append(CLASS_MAP[r["result"]])
        prob_rows.append(prob_vec)
        predictor.observe(r["home_team"], r["away_team"],
                          float(r["home_goals"]), float(r["away_goals"]),
                          r["result"], prob_vec=prob_vec)
    y_true = np.array(y_true)
    probs = np.array(prob_rows)
    pred = np.argmax(probs, axis=1)
    from sklearn.metrics import balanced_accuracy_score
    half = len(y_true) // 2
    acc_first = float(np.mean(pred[:half] == y_true[:half])) if half else float("nan")
    acc_second = float(np.mean(pred[half:] == y_true[half:])) if half else float("nan")
    return {
        "accuracy": round(float(np.mean(pred == y_true)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "log_loss": round(float(-np.mean(np.log(np.clip(
            probs[np.arange(len(y_true)), y_true], 1e-9, 1)))), 4),
        "brier": round(brier(y_true, probs), 4),
        "acc_first_half": round(acc_first, 4),
        "acc_second_half": round(acc_second, 4),
        "refits": predictor.refits,
    }


def majority_acc(y: np.ndarray) -> float:
    return float(np.max(np.bincount(y, minlength=3)) / len(y))


def load_train(league: str) -> pd.DataFrame:
    parts = [get_season(league, s) for s in TRAIN_SEASONS]
    return pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)


def synthetic_other_sport(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """A basketball-like world: high scoring, draws almost never happen."""
    rng = np.random.default_rng(seed)
    teams = [f"BK{i}" for i in range(10)]
    strength = {t: float(rng.normal(0, 1)) for t in teams}
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    home, away = [], []
    for _ in range(n):
        h = str(rng.choice(teams))
        a = str(rng.choice([t for t in teams if t != h]))
        home.append(h); away.append(a)
    diff = np.array([strength[h] - strength[a] for h, a in zip(home, away)])
    lam_h = 105 + 14 * diff          # basketball scoring scale
    lam_a = 100 - 14 * diff
    hg = rng.normal(lam_h, 12).round().clip(0, None)
    ag = rng.normal(lam_a, 12).round().clip(0, None)
    result = np.where(hg > ag, "H", np.where(hg < ag, "A", "H"))  # OT -> home
    return pd.DataFrame({"date": dates, "home_team": home, "away_team": away,
                         "home_goals": hg, "away_goals": ag, "result": result})


def run_experiment(name: str, train: pd.DataFrame, test: pd.DataFrame,
                   results: list) -> None:
    y_test = test["result"].map(CLASS_MAP).to_numpy()
    base_acc = majority_acc(y_test)
    rows = {}

    for static in (False, True):
        model = AdaptiveMatchPredictor(static=static)
        model.train(train)
        met = walk(model, test)
        rows["Adaptive" if not static else "Static (frozen ML)"] = met
        rows["Adaptive" if not static else "Static (frozen ML)"].update(
            {"base_acc": base_acc})

    print(f"\n  {name}  ({len(test)} unseen matches, majority acc {base_acc:.1%})")
    print(f"  {'Method':<22}{'Acc':>7}{'BalAcc':>8}{'LogLoss':>10}{'Brier':>9}{'Refits':>7}")
    print(f"  {'':<22}{'1st/2nd':>7}{'':>8}{'':>10}{'':>9}{'':>7}")
    for method, m in rows.items():
        print(f"  {method:<22}{m['accuracy']:>7.1%}{m['balanced_acc']:>8.1%}"
              f"{m['log_loss']:>10.3f}{m['brier']:>9.3f}{m['refits']:>7}"
              f"  1st {m['acc_first_half']:.1%} / 2nd {m['acc_second_half']:.1%}")
        results.append({"experiment": name, "method": method, "majority_acc": base_acc, **m})


def main():
    parser = argparse.ArgumentParser(description="Adaptive cross-league transfer")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("ADAPTIVE CROSS-LEAGUE / CROSS-SPORT TRANSFER (point-in-time)")
    print("A model trained on ONE league is pointed at unseen leagues AND a")
    print("different sport - does online adaptation beat a frozen model?")
    print("=" * 78)

    results = []

    # --- A/B/C: train on Serie A, test on Serie A / La Liga / EPL 2025/26
    print("\n[1/2] REAL FOOTBALL — trained on Serie A 2020/21-2023/24")
    serie_a = load_train("I1")
    serie_a_2526 = get_season("I1", TEST_SEASON)
    la_liga_2526 = get_season("SP1", TEST_SEASON)
    epl_2526 = get_season("E0", TEST_SEASON)

    run_experiment("Serie A -> Serie A 25/26 (within-league unseen season)",
                   serie_a, serie_a_2526, results)
    run_experiment("Serie A -> La Liga 25/26 (cross-league transfer)",
                   serie_a, la_liga_2526, results)
    run_experiment("Serie A -> Premier League 25/26 (cross-league transfer)",
                   serie_a, epl_2526, results)

    # --- D: cross-sport
    print("\n[2/2] CROSS-SPORT — football-trained model on a basketball-like world")
    sport_train = serie_a
    sport_test = synthetic_other_sport()
    run_experiment("Football -> synthetic basketball-like league (cross-sport)",
                   sport_train, sport_test, results)

    # --- save
    res = pd.DataFrame(results)
    res.to_csv(RESULTS_DIR / "adaptive_transfer_results.csv", index=False)
    print(f"\n[OK] Saved {RESULTS_DIR / 'adaptive_transfer_results.csv'}")

    _write_doc(res)


def _write_doc(res: pd.DataFrame):
    lines = [
        "# Adaptive Cross-League / Cross-Sport Transfer",
        "",
        "One model is trained on **real Serie A 2020/21-2023/24** and then pointed at",
        "unseen matches in Serie A, La Liga, the Premier League (all 2025/26) and a",
        "synthetic basketball-like league. Every prediction uses only information",
        "known before kick-off; after each match the online state (Elo, form) updates",
        "and the **adaptive** model optionally refits its ML layer on a rolling window",
        "when scheduled or when its rolling Brier drifts. The **static** control never",
        "refits, so the gap isolates the value of adaptation.",
        "",
        "```",
        res.to_string(index=False),
        "```",
        "",
        "*(Saved by `scripts/10_adaptive_transfer.py`; full numbers in",
        "`backtests/results/adaptive_transfer_results.csv`.)*",
    ]
    doc = PROJECT_ROOT / "docs" / "10_adaptive_transfer.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {doc}")


if __name__ == "__main__":
    main()
