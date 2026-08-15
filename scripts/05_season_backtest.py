#!/usr/bin/env python3
"""
Season-by-Season Backtest on REAL league data (time-aware validation).

This is the project's real-data evaluation: instead of a random split, every
test prediction is a genuinely unseen FUTURE match.

Experiment 1 - WITHIN-LEAGUE, expanding window (La Liga):
    train 2021/22            -> test 2022/23
    train 2021/22..2022/23   -> test 2023/24
    train 2021/22..2023/24   -> test 2024/25
    train 2021/22..2024/25   -> test 2025/26

Experiment 2 - CROSS-LEAGUE transfer:
    train all La Liga  -> test Premier League 2025/26 (unseen matches)
    train all EPL      -> test La Liga 2025/26 (unseen matches)

Methodology
-----------
* Features are computed ONLINE: for match *i* they use only matches strictly
  before *i* (running Elo ratings + last-5 rolling form).  No future leakage.
* Teams unseen in training get the Elo base (1500) and league-average form, so
  promoted teams / new leagues are a genuine cold start.
* Models: majority baseline, the project's PoissonElo model, Ridge, Gradient
  Boosting, Random Forest (all calibrated via CalibratedClassifierCV).
* Probability models are scored with accuracy, balanced accuracy, log-loss and
  Brier; calibration is checked with the expected calibration error (ECE).

Results are saved to backtests/results/season_backtest_results.csv and
backtests/results/season_backtest_*.png, and summarised in
docs/05_season_backtest.md.

Usage:
    python scripts/05_season_backtest.py            # downloads real data
    python scripts/05_season_backtest.py --offline  # use cached data
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

from models.poisson_elo_model import PoissonEloModel  # noqa: E402

REAL_DIR = PROJECT_ROOT / "data" / "real"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
SEASON_LABEL = {"2122": "2021/22", "2223": "2022/23", "2324": "2023/24",
                "2425": "2024/25", "2526": "2025/26"}
LEAGUES = {"SP1": "La Liga", "E0": "Premier League"}

CLASS_MAP = {"A": 0, "D": 1, "H": 2}
OUTCOME_NAMES = ["away_win", "draw", "home_win"]
FEATURE_COLS = ["home_elo", "away_elo", "home_goals_avg", "away_goals_avg"]

ELO_K = 20.0
ELO_BASE = 1500.0
BASE_HOME_GOALS = 1.6
BASE_AWAY_GOALS = 1.3


# ------------------------------------------------------------- data loading
def download_season(league: str, season: str) -> pd.DataFrame:
    """Download one real season, normalised to the project schema."""
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    })
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["league"] = LEAGUES[league]
    df["season"] = SEASON_LABEL[season]
    keep = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "result", "league", "season"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["home_goals", "away_goals", "result"])
    return df.reset_index(drop=True)


def get_season(league: str, season: str, offline: bool) -> pd.DataFrame:
    cache = REAL_DIR / f"{league}_{season}.csv"
    if offline:
        if not cache.exists():
            sys.exit(f"[FAIL] --offline but {cache} missing. Run once without "
                     f"--offline to download.")
    else:
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            cache.write_bytes(pd.read_csv(
                f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            ).to_csv(index=False).encode())
    return download_season(league, season)


# ------------------------------------------------------ online feature builder
class OnlineFeatureBuilder:
    """Running Elo + last-5 rolling form; features for row i use rows < i only."""

    def __init__(self):
        self.elo = defaultdict(lambda: ELO_BASE)
        self.home_form: dict = defaultdict(list)  # goals scored at home
        self.away_form: dict = defaultdict(list)  # goals conceded away

    @staticmethod
    def _avg(xs, default: float) -> float:
        return float(np.mean(xs[-5:])) if xs else default

    def transform(self, df: pd.DataFrame, update: bool = True) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            h, a = r["home_team"], r["away_team"]
            rows.append({
                "home_elo": float(self.elo[h]),
                "away_elo": float(self.elo[a]),
                "home_goals_avg": self._avg(self.home_form[h], BASE_HOME_GOALS),
                "away_goals_avg": self._avg(self.away_form[a], BASE_AWAY_GOALS),
            })
            if update:
                self._update(r, h, a)
        return pd.DataFrame(rows)

    def _update(self, r, h, a):
        home_rating, away_rating = self.elo[h], self.elo[a]
        exp_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        actual = 1.0 if r["home_goals"] > r["away_goals"] else (
            0.0 if r["home_goals"] < r["away_goals"] else 0.5)
        self.elo[h] += ELO_K * (actual - exp_home)
        self.elo[a] += ELO_K * ((1 - actual) - (1 - exp_home))
        self.home_form[h].append(float(r["home_goals"]))
        self.away_form[a].append(float(r["away_goals"]))


# --------------------------------------------------------------- evaluation
def evaluate(y_true: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    eps = 1e-9
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y_true))
    bacc = float(balanced_accuracy_score(y_true, pred))
    ll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], eps, 1))))
    brier = float(np.mean(np.sum((probs - np.eye(3)[y_true]) ** 2, axis=1)))
    ece = _expected_calibration_error(y_true, probs)
    return {"accuracy": round(acc, 4), "balanced_acc": round(bacc, 4),
            "log_loss": round(ll, 4), "brier": round(brier, 4), "ece": round(ece, 4)}


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                                n_bins: int = 10) -> float:
    """Class-agnostic ECE: confidence (max prob) vs accuracy in that bin."""
    conf = probs.max(axis=1)
    correct = (np.argmax(probs, axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - conf[mask].mean())
    return ece


def majority_probs(y_train: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=3)
    base = counts / counts.sum()
    return np.tile(base, (n, 1))


# ------------------------------------------------------------------- models
def fit_sklearn(clf, X: pd.DataFrame, y: np.ndarray):
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(clf, method="sigmoid", cv=3).fit(X, y)


def poisson_elo_sequential(train: pd.DataFrame, test: pd.DataFrame,
                           y_test: np.ndarray) -> np.ndarray:
    """Project's PoissonElo model: fit on train, predict test match by match,
    updating Elo ratings online (predictions only use past matches)."""
    model = PoissonEloModel(elo_k=ELO_K)
    model.train(train)
    probs = []
    for _, r in test.iterrows():
        p = model.predict(r["home_team"], r["away_team"])
        probs.append([p["away_win"], p["draw"], p["home_win"]])
        model._update_elo(r["home_team"], r["away_team"],
                          int(r["home_goals"]), int(r["away_goals"]))
    return np.array(probs, dtype=float)


def run_experiment(name: str, train: pd.DataFrame, test: pd.DataFrame,
                   results: list, verbose: bool = True) -> None:
    """Fit on train (features computed online), predict on test sequentially."""
    y_train = train["result"].map(CLASS_MAP).to_numpy()
    y_test = test["result"].map(CLASS_MAP).to_numpy()

    # --- baselines
    rows = {"Majority / base rate": majority_probs(y_train, len(test))}

    # --- PoissonElo (project core model)
    rows["PoissonElo model"] = poisson_elo_sequential(train, test, y_test)

    # --- sklearn models on online features
    builder = OnlineFeatureBuilder()
    X_train = builder.transform(train)
    X_test = builder.transform(test)

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import RidgeClassifier

    X_train = X_train.fillna({c: 0.0 for c in FEATURE_COLS})
    X_test = X_test.fillna({c: 0.0 for c in FEATURE_COLS})

    ridge = fit_sklearn(RidgeClassifier(alpha=1.0), X_train, y_train)
    rows["Ridge classifier"] = ridge.predict_proba(X_test)

    gb = fit_sklearn(GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=42), X_train, y_train)
    rows["Gradient Boosting"] = gb.predict_proba(X_test)

    rf = fit_sklearn(RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42),
        X_train, y_train)
    rows["Random Forest"] = rf.predict_proba(X_test)

    if verbose:
        print(f"\n  {name}  ({len(test)} matches)")
        print(f"  {'Method':<24}{'Acc':>7}{'BalAcc':>8}{'LogLoss':>10}{'Brier':>9}{'ECE':>7}")
    for method, probs in rows.items():
        ev = evaluate(y_test, probs)
        results.append({"experiment": name, "method": method, **ev})
        if verbose:
            print(f"  {method:<24}{ev['accuracy']:>7.1%}{ev['balanced_acc']:>8.1%}"
                  f"{ev['log_loss']:>10.3f}{ev['brier']:>9.3f}{ev['ece']:>7.3f}")


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Season-by-season real-data backtest")
    parser.add_argument("--offline", action="store_true",
                        help="use cached data instead of downloading")
    args = parser.parse_args()

    print("=" * 78)
    print("SEASON-BY-SEASON BACKTEST ON REAL LEAGUE DATA (time-aware validation)")
    print("Every test prediction is a genuinely unseen future match.")
    print("=" * 78)

    results = []

    # ---- Exp 1: within-league expanding window on La Liga
    print("\n[1/2] WITHIN-LEAGUE (La Liga) - expanding window")
    train_parts = []
    for i, season in enumerate(SEASONS[1:], start=1):
        train_parts.append(get_season("SP1", SEASONS[i - 1], args.offline))
        train = pd.concat(train_parts, ignore_index=True).sort_values("date")
        test = get_season("SP1", season, args.offline)
        run_experiment(f"La Liga: train {SEASON_LABEL[SEASONS[0]]}-{SEASON_LABEL[SEASONS[i - 1]]} -> test {SEASON_LABEL[season]}",
                       train, test, results)

    # ---- Exp 2: cross-league transfer
    print("\n[2/2] CROSS-LEAGUE TRANSFER")
    la_liga = pd.concat([get_season("SP1", s, args.offline) for s in SEASONS[:-1]],
                        ignore_index=True).sort_values("date")
    epl = pd.concat([get_season("E0", s, args.offline) for s in SEASONS[:-1]],
                    ignore_index=True).sort_values("date")
    la_liga_2526 = get_season("SP1", "2526", args.offline)
    epl_2526 = get_season("E0", "2526", args.offline)

    run_experiment("La Liga 21/22-24/25 -> Premier League 25/26 (unseen)", la_liga, epl_2526, results)
    run_experiment("Premier League 21/22-24/25 -> La Liga 25/26 (unseen)", epl, la_liga_2526, results)

    # ---- save
    res = pd.DataFrame(results)
    res.to_csv(RESULTS_DIR / "season_backtest_results.csv", index=False)
    print(f"\n[OK] Saved {RESULTS_DIR / 'season_backtest_results.csv'}")

    _plot_season(res)
    _write_summary_doc(res)


def _plot_season(res: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    within = res[res["experiment"].str.startswith("La Liga: train")]
    # aggregate per season (first 4 characters of the test season label)
    seasons = []
    for _, row in within.iterrows():
        seasons.append(row["experiment"].split("-> test ")[-1])
    within = within.copy()
    within["test_season"] = seasons
    order = ["Majority / base rate", "PoissonElo model", "Ridge classifier",
             "Gradient Boosting", "Random Forest"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "accuracy", "Within-league accuracy by season (real La Liga)"),
        (axes[1], "log_loss", "Within-league log-loss by season"),
    ]:
        for method in order:
            sub = within[within["method"] == method].sort_values("test_season")
            ax.plot(sub["test_season"], sub[metric], marker="o", label=method)
        ax.set_xlabel("Test season (expanding-window train)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "season_backtest_within.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    cross = res[~res["experiment"].str.startswith("La Liga: train")]
    fig, ax = plt.subplots(figsize=(11, 5))
    methods = [m for m in order if m in cross["method"].values]
    x = np.arange(len(methods))
    width = 0.18
    for j, exp in enumerate(cross["experiment"].unique()):
        vals = [cross[(cross["experiment"] == exp) & (cross["method"] == m)]["accuracy"].iloc[0]
                for m in methods]
        ax.bar(x + (j - 0.5) * width, np.array(vals) * 100, width, label=exp[:42])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross-league transfer accuracy (models trained on real data)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "season_backtest_cross.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved season_backtest_within.png / season_backtest_cross.png")


def _write_summary_doc(res: pd.DataFrame):
    lines = [
        "# Season-by-Season Backtest on Real League Data",
        "",
        "Time-aware validation on **real** match data (football-data.co.uk). Every test",
        "prediction is a genuinely unseen future match; features for match *i* use only",
        "matches strictly before *i* (running Elo + last-5 form).",
        "",
        "## Within-league (La Liga, expanding window)",
        "",
        "```",
        res[res["experiment"].str.startswith("La Liga: train")].to_string(index=False),
        "```",
        "",
        "## Cross-league transfer",
        "",
        "```",
        res[~res["experiment"].str.startswith("La Liga: train")].to_string(index=False),
        "```",
        "",
        "*(Saved by `scripts/05_season_backtest.py`; full numbers in",
        "`backtests/results/season_backtest_results.csv`.)*",
    ]
    doc = PROJECT_ROOT / "docs" / "05_season_backtest.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {doc}")


if __name__ == "__main__":
    main()
