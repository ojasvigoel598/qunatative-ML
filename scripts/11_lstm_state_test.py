#!/usr/bin/env python3
"""
STATEFUL SEQUENCE MODELS (LSTM / GRU) vs BASELINES — with a database-vs-model test.

Two questions
------------
Q1. Does an architecture that models *changing state* (an LSTM/GRU over each
    team's rolling match sequence) beat feed-forward baselines on real,
    previously-unseen matches?

Q2. (Your thesis, tested directly) Does the DATABASE matter more than the
    model?  We train the SAME architecture on 1 vs 2 vs 3 real leagues, and
    the SAME league with thin (goals-only) vs rich (+shots/corners/cards/odds)
    features, and measure which lever moves accuracy more.

Protocol (identical for every method)
-------------------------------------
* Train on real matches strictly BEFORE the test window (Serie A 2020/21-
  2023/24 -> test 2025/26, and cross-league into La Liga / EPL 2025/26).
* Walk the test chronologically: predict -> record -> reveal -> update the
  online state (histories / Elo / form).  Zero future information anywhere.
* No test set is ever used for tuning; LSTM early-stops on a chronological
  validation slice cut from the training window.

Methods
-------
* Majority / base rate
* PoissonElo (project core)
* Gradient Boosting on online features (CalibratedClassifierCV)
* Adaptive model (online refits)
* LSTM over rolling per-team sequences (rich features)
* GRU   over rolling per-team sequences (rich features)
* LSTM thin: goals-only features (isolates the value of the extra database)
* LSTM with NO pre-match odds in the static vector (isolates market info)

Results -> backtests/results/lstm_state_results.csv + docs/11_lstm_state_test.md

Usage:
    python scripts/11_lstm_state_test.py --offline
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

from data.real_data import load_league_rich, LEAGUES, SEASON_LABEL  # noqa: E402
from models.adaptive_model import CLASS_MAP, ELO_BASE, OnlineState, brier  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = ["2021", "2122", "2223", "2324"]   # 2020/21 .. 2023/24
TEST_SEASON = "2526"


def evaluate(y_true: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score
    y_true = np.asarray(y_true); probs = np.asarray(probs)
    pred = np.argmax(probs, axis=1)
    eps = 1e-9
    return {
        "accuracy": round(float(np.mean(pred == y_true)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "log_loss": round(float(-np.mean(np.log(np.clip(
            probs[np.arange(len(y_true)), y_true], eps, 1)))), 4),
        "brier": round(brier(y_true, probs), 4),
    }


def _odds_from_row(r) -> dict:
    return {"home_win": r.get("odds_home"), "draw": r.get("odds_draw"),
            "away_win": r.get("odds_away")}


# ------------------------------------------------------------------- baselines
def majority_probs(y_train: np.ndarray, n: int) -> np.ndarray:
    c = np.bincount(y_train, minlength=3)
    return np.tile(c / c.sum(), (n, 1))


def poisson_elo_walk(train: pd.DataFrame, test: pd.DataFrame):
    from models.poisson_elo_model import PoissonEloModel
    model = PoissonEloModel(elo_k=20.0)
    model.train(train)
    probs = []
    for _, r in test.iterrows():
        p = model.predict(r["home_team"], r["away_team"])
        probs.append([p["away_win"], p["draw"], p["home_win"]])
        model._update_elo(r["home_team"], r["away_team"],
                          int(r["home_goals"]), int(r["away_goals"]))
    return np.array(probs, dtype=float)


def gb_walk(train: pd.DataFrame, test: pd.DataFrame):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    st = OnlineState()
    X, y = [], []
    for _, r in train.iterrows():
        f = st.features(r["home_team"], r["away_team"])
        X.append([f["elo_diff"], f["home_goals_avg"], f["away_goals_avg"],
                  f["home_pts_5"], f["away_pts_5"]])
        y.append(CLASS_MAP[r["result"]])
        st.update(r["home_team"], r["away_team"], float(r["home_goals"]),
                  float(r["away_goals"]), r["result"])
    clf = CalibratedClassifierCV(GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05,
        min_samples_leaf=15, subsample=0.8, random_state=42),
        method="sigmoid", cv=3).fit(np.array(X), np.array(y))
    probs = []
    for _, r in test.iterrows():
        f = st.features(r["home_team"], r["away_team"])
        x = np.array([[f["elo_diff"], f["home_goals_avg"], f["away_goals_avg"],
                       f["home_pts_5"], f["away_pts_5"]]])
        probs.append(clf.predict_proba(x)[0])
        st.update(r["home_team"], r["away_team"], float(r["home_goals"]),
                  float(r["away_goals"]), r["result"])
    return np.array(probs, dtype=float)


def adaptive_walk(train: pd.DataFrame, test: pd.DataFrame):
    from models.adaptive_model import AdaptiveMatchPredictor
    model = AdaptiveMatchPredictor(static=False)
    model.train(train)
    probs = []
    for _, r in test.iterrows():
        p = model.predict(r["home_team"], r["away_team"])
        probs.append([p["away_win"], p["draw"], p["home_win"]])
        model.observe(r["home_team"], r["away_team"], float(r["home_goals"]),
                      float(r["away_goals"]), r["result"])
    return np.array(probs, dtype=float)


def lstm_walk(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
              cell: str, rich: bool, use_odds: bool):
    from models.lstm_model import StatefulSequenceModel, STATIC_DIM
    m = StatefulSequenceModel(rich=rich, cell=cell, epochs=45)
    m.train(train, valid)
    probs = []
    for _, r in test.iterrows():
        odds = _odds_from_row(r) if use_odds else None
        p = m.predict(r["home_team"], r["away_team"], odds)
        probs.append([p["away_win"], p["draw"], p["home_win"]])
        m.observe(r["home_team"], r["away_team"], float(r["home_goals"]),
                  float(r["away_goals"]), r["result"], row=r)
    return np.array(probs, dtype=float)


# --------------------------------------------------------------------- runner
def run_experiment(name: str, train: pd.DataFrame, test: pd.DataFrame,
                   results: list, cells=("lstm", "gru")) -> None:
    """train: rich df (already chrono).  valid = last 12% of train, chrono."""
    y_train = train["result"].map(CLASS_MAP).to_numpy()
    y_test = test["result"].map(CLASS_MAP).to_numpy()
    n_v = max(int(len(train) * 0.12), 60)
    valid, train_fit = train.iloc[-n_v:], train.iloc[:-n_v]

    rows = {"Majority / base rate": majority_probs(y_train, len(test))}
    rows["PoissonElo model"] = poisson_elo_walk(train_fit, test)
    rows["Gradient Boosting"] = gb_walk(train_fit, test)
    rows["Adaptive (online refits)"] = adaptive_walk(train_fit, test)

    for cell in cells:
        rows[f"{cell.upper()} (rich)"] = lstm_walk(train_fit, valid, test, cell,
                                                   rich=True, use_odds=True)
    rows["LSTM thin (goals only)"] = lstm_walk(train_fit, valid, test, "lstm",
                                               rich=False, use_odds=True)
    rows["LSTM rich, NO odds"] = lstm_walk(train_fit, valid, test, "lstm",
                                           rich=True, use_odds=False)

    print(f"\n  {name}  ({len(test)} unseen matches)")
    print(f"  {'Method':<26}{'Acc':>7}{'BalAcc':>8}{'LogLoss':>10}{'Brier':>9}")
    for method, probs in rows.items():
        ev = evaluate(y_test, probs)
        print(f"  {method:<26}{ev['accuracy']:>7.1%}{ev['balanced_acc']:>8.1%}"
              f"{ev['log_loss']:>10.3f}{ev['brier']:>9.3f}")
        results.append({"experiment": name, "method": method, **ev})


def run_db_experiment(results: list):
    """Database-vs-model: SAME LSTM on 1 vs 2 vs 3 leagues; test Serie A 25/26."""
    serie_a = load_league_rich("I1", TRAIN_SEASONS)
    la_liga = load_league_rich("SP1", TRAIN_SEASONS)
    epl = load_league_rich("E0", TRAIN_SEASONS)
    test = load_league_rich("I1", [TEST_SEASON])
    y_test = test["result"].map(CLASS_MAP).to_numpy()

    dbs = {
        "1 league (Serie A only)": serie_a,
        "2 leagues (+ La Liga)": pd.concat([serie_a, la_liga], ignore_index=True),
        "3 leagues (+ EPL)": pd.concat([serie_a, la_liga, epl], ignore_index=True),
    }
    print("\n" + "=" * 78)
    print("DATABASE vs MODEL — same LSTM, bigger database, same unseen test")
    print("(test = Serie A 2025/26 in every row)")
    print("=" * 78)
    print(f"  {'Database':<26}{'Acc':>7}{'BalAcc':>8}{'LogLoss':>10}{'Brier':>9}")
    for label, db in dbs.items():
        db = db.sort_values("date").reset_index(drop=True)
        n_v = max(int(len(db) * 0.12), 60)
        valid, train_fit = db.iloc[-n_v:], db.iloc[:-n_v]
        probs = lstm_walk(train_fit, valid, test, "lstm", rich=True, use_odds=True)
        ev = evaluate(y_test, probs)
        print(f"  {label:<26}{ev['accuracy']:>7.1%}{ev['balanced_acc']:>8.1%}"
              f"{ev['log_loss']:>10.3f}{ev['brier']:>9.3f}")
        results.append({"experiment": f"DB-size: {label} -> Serie A 25/26",
                        "method": "LSTM (rich)", **ev})


def main():
    parser = argparse.ArgumentParser(description="LSTM/GRU state-space test")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-db", action="store_true", help="skip the DB-size experiment")
    args = parser.parse_args()

    print("=" * 78)
    print("STATEFUL SEQUENCE MODELS (LSTM / GRU) vs BASELINES — real data")
    print("trained on real Serie A 2020/21-2023/24, tested on unseen 2025/26")
    print("=" * 78)

    results = []
    serie_a = load_league_rich("I1", TRAIN_SEASONS)
    serie_a_2526 = load_league_rich("I1", [TEST_SEASON])
    la_liga_2526 = load_league_rich("SP1", [TEST_SEASON])
    epl_2526 = load_league_rich("E0", [TEST_SEASON])

    run_experiment("Serie A -> Serie A 25/26 (within-league)", serie_a, serie_a_2526, results)
    run_experiment("Serie A -> La Liga 25/26 (cross-league)", serie_a, la_liga_2526, results)
    run_experiment("Serie A -> Premier League 25/26 (cross-league)", serie_a, epl_2526, results)

    if not args.skip_db:
        run_db_experiment(results)

    res = pd.DataFrame(results)
    res.to_csv(RESULTS_DIR / "lstm_state_results.csv", index=False)
    print(f"\n[OK] Saved {RESULTS_DIR / 'lstm_state_results.csv'}")

    _write_doc(res)


def _write_doc(res: pd.DataFrame):
    lines = [
        "# Stateful Sequence Models (LSTM / GRU) on Real Data",
        "",
        "An LSTM/GRU over each team's rolling last-8 match sequence — the hidden",
        "state IS the team's learned evolving form — compared with the feed-forward",
        "baselines under the identical point-in-time protocol (train strictly before",
        "the test window, predict -> reveal -> update online, zero future info).",
        "",
        "## Head-to-head on unseen 2025/26 matches (trained on Serie A 2020/21-2023/24)",
        "",
        "```",
        res[res["experiment"].str.startswith("Serie A")].to_string(index=False),
        "```",
        "",
        "## Database vs model (your thesis, tested directly)",
        "",
        "The SAME LSTM is trained on 1, 2 and 3 real leagues; the test is Serie A",
        "2025/26 in every row. If the database matters more than the architecture,",
        "accuracy should rise with database size.",
        "",
        "```",
        res[res["experiment"].str.startswith("DB-size")].to_string(index=False),
        "```",
        "",
        "*(Saved by `scripts/11_lstm_state_test.py`; full numbers in",
        "`backtests/results/lstm_state_results.csv`.)*",
    ]
    doc = PROJECT_ROOT / "docs" / "11_lstm_state_test.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {doc}")


if __name__ == "__main__":
    main()
