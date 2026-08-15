#!/usr/bin/env python3
"""
Deep-Learning Transfer Experiment.

Trains a **PyTorch neural network** and a **TensorFlow hybrid model**
(PoissonElo outputs fused into an MLP) on ALL synthetic training data, then
answers two transfer questions:

  1. CROSS-LEAGUE : does a model trained on synthetic data work on the latest
                    real La Liga season (2025/26)?
  2. OUT-OF-SAMPLE: does it work on real Premier League matches it has never
                    seen (2025/26)?

Features for the real leagues are computed from the *previous* real season
(2024/25) - Elo ratings and rolling form - so the transfer test is about the
learned FEATURE->PROBABILITY mapping, not about cold-starting on unknown teams.
A cold-start row (no team information at all) is reported separately.

Baselines: the most-common-outcome base rate and the bookmaker's own implied
probabilities (the real market).  Results are saved to
backtests/results/transfer_* and summarised in docs/04_deep_learning_transfer.md.

Usage:
    python scripts/04_deep_learning_transfer.py            # downloads real data
    python scripts/04_deep_learning_transfer.py --offline  # use cached data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pipeline  # noqa: E402
from models.ml_layer import MLFootballPredictor  # noqa: E402
from models.nn_model import NNFootballPredictor  # noqa: E402
from models.poisson_elo_model import PoissonEloModel  # noqa: E402
from models.tf_hybrid import TFHybridPredictor  # noqa: E402

REAL_DIR = PROJECT_ROOT / "data" / "real"
RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = {"2324": "2023/24", "2425": "2024/25", "2526": "2025/26"}
LEAGUES = {"SP1": "La Liga", "E0": "Premier League"}
FEATURE_SEASON = "2425"   # real features come from the previous season
TEST_SEASON = "2526"      # the latest season - genuinely unseen matches

CLASS_MAP = {"A": 0, "D": 1, "H": 2}
OUTCOME_NAMES = ["away_win", "draw", "home_win"]
FEATURE_COLS = ["home_elo", "away_elo", "home_goals_avg", "away_goals_avg"]


# --------------------------------------------------------------------- data
def download_real_season(league: str, season: str) -> pd.DataFrame:
    """Download one real season CSV and normalise columns to the project schema."""
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "B365H": "odds_home_b365", "B365D": "odds_draw_b365", "B365A": "odds_away_b365",
    })
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["league"] = LEAGUES[league]
    df["season"] = SEASONS[season]
    keep = ["date", "home_team", "away_team", "home_goals", "away_goals", "result",
            "odds_home_b365", "odds_draw_b365", "odds_away_b365", "league", "season"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["home_goals", "away_goals", "result"])
    return df.reset_index(drop=True)


def get_real_data(league: str, offline: bool) -> dict:
    """Return {'features': 2425 df, 'test': 2526 df}, downloading if needed."""
    path_feat = REAL_DIR / f"{league}_{FEATURE_SEASON}.csv"
    path_test = REAL_DIR / f"{league}_{TEST_SEASON}.csv"
    if offline:
        if not (path_feat.exists() and path_test.exists()):
            sys.exit(f"[FAIL] --offline but {league} data missing in {REAL_DIR}. "
                     f"Run once without --offline to download.")
    else:
        REAL_DIR.mkdir(parents=True, exist_ok=True)
        path_feat.parent.mkdir(parents=True, exist_ok=True)
        path_feat.write_bytes(pd.read_csv(
            f"https://www.football-data.co.uk/mmz4281/{FEATURE_SEASON}/{league}.csv"
        ).to_csv(index=False).encode())
        path_test.write_bytes(pd.read_csv(
            f"https://www.football-data.co.uk/mmz4281/{TEST_SEASON}/{league}.csv"
        ).to_csv(index=False).encode())
    feat = download_real_season(league, FEATURE_SEASON)
    test = download_real_season(league, TEST_SEASON)
    return {"features": feat, "test": test}


def synthetic_features() -> tuple:
    """Full synthetic training set: features X, y, poisson probs P, poisson model."""
    df, _ = pipeline.load_or_generate_data(n_matches=1200, seed=42)
    feat = NNFootballPredictor.build_features(df)  # Elo + shifted form
    X = feat[FEATURE_COLS].copy()
    X["home_elo"] = X["home_elo"].fillna(1500.0)
    X["away_elo"] = X["away_elo"].fillna(1500.0)
    X["home_goals_avg"] = X["home_goals_avg"].fillna(1.6)
    X["away_goals_avg"] = X["away_goals_avg"].fillna(1.3)
    y = df["result"].map(CLASS_MAP).to_numpy()

    poisson = PoissonEloModel()
    poisson.train(df)
    P = TFHybridPredictor.build_poisson_probs(poisson, df)
    return X.to_numpy(dtype=np.float32), y, P, poisson, df


def real_features(df: pd.DataFrame) -> np.ndarray:
    """Features for real matches computed from that league's own prior season."""
    feat = NNFootballPredictor.build_features(df)
    X = feat[FEATURE_COLS].copy()
    X["home_elo"] = X["home_elo"].fillna(1500.0)
    X["away_elo"] = X["away_elo"].fillna(1500.0)
    X["home_goals_avg"] = X["home_goals_avg"].fillna(1.6)
    X["away_goals_avg"] = X["away_goals_avg"].fillna(1.3)
    return X.to_numpy(dtype=np.float32)


def real_poisson_probs(poisson, df: pd.DataFrame) -> np.ndarray:
    """PoissonElo probs for real teams (base-rate for unseen teams)."""
    return TFHybridPredictor.build_poisson_probs(poisson, df)


# --------------------------------------------------------------- evaluation
def evaluate(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    eps = 1e-9
    acc = float(np.mean(np.argmax(probs, axis=1) == y_true))
    ll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], eps, 1))))
    brier = float(np.mean(np.sum((probs - np.eye(3)[y_true]) ** 2, axis=1)))
    return {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "brier": round(brier, 4)}


def market_probs(df: pd.DataFrame) -> np.ndarray:
    """Bookmaker-implied probabilities (normalised inverse odds).

    NOTE: the odds must be inverted (1/odds) before normalising; the original
    code normalised the raw odds, which made "argmax" pick the bookmaker's
    *least* likely outcome and inverted the market baseline.
    """
    odds = df[["odds_away_b365", "odds_draw_b365", "odds_home_b365"]].to_numpy(dtype=float)
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def base_rate_probs(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3)
    base = counts / counts.sum()
    return np.tile(base, (len(y), 1))


def fit_sklearn_baseline(clf, X: np.ndarray, y: np.ndarray) -> object:
    """Fit + sigmoid-calibrate a sklearn classifier (consistent with GB)."""
    cal = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    cal.fit(X, y)
    return cal


def run_league(models: dict, league: str, offline: bool, results: list,
               poisson) -> pd.DataFrame:
    print(f"\n{'=' * 78}\nLEAGUE TEST: {LEAGUES[league]} 2025/26 "
          f"(features from 2024/25, {TEST_SEASON} matches unseen)\n{'=' * 78}")
    data = get_real_data(league, offline)
    test = data["test"]
    y_true = test["result"].map(CLASS_MAP).to_numpy()

    X_real = real_features(data["features"])              # trained on 2425
    P_real = real_poisson_probs(poisson, data["features"])

    print(f"  Matches: {len(test)} | home-win rate {np.mean(test['result'] == 'H'):.2%} "
          f"| draw rate {np.mean(test['result'] == 'D'):.2%}")

    # Baselines
    base = base_rate_probs(y_true)
    market = market_probs(test)
    rows = {"Base rate (most common)": base, "Market (bookmaker odds)": market}

    # Models (trained on synthetic data only)
    X_real_df = pd.DataFrame(X_real, columns=FEATURE_COLS)
    rows["PyTorch NN"] = models["nn"].predict_proba_matrix(X_real)
    rows["TF hybrid (NN + PoissonElo)"] = models["tf"].predict_proba_matrix(X_real, P_real)
    rows["sklearn Gradient Boosting"] = models["gb"].predict_proba_matrix(X_real_df)
    rows["sklearn Logistic Regression"] = models["lr"].predict_proba(X_real_df)
    rows["sklearn Ridge classifier"] = models["ridge"].predict_proba(X_real_df)
    rows["sklearn Random Forest"] = models["rf"].predict_proba(X_real_df)

    # Cold-start: no team information at all (the null transfer)
    cold_X = np.tile(np.array([[1500.0, 1500.0, 1.6, 1.3]], dtype=np.float32), (len(test), 1))
    rows["PyTorch NN - cold start (no team info)"] = models["nn"].predict_proba_matrix(cold_X)

    print(f"\n  {'Method':<34}{'Acc':>7}{'LogLoss':>10}{'Brier':>9}")
    for name, probs in rows.items():
        ev = evaluate(y_true, probs)
        results.append({"league": LEAGUES[league], "method": name, **ev})
        print(f"  {name:<34}{ev['accuracy']:>7.1%}{ev['log_loss']:>10.3f}{ev['brier']:>9.3f}")

    return test


def main():
    parser = argparse.ArgumentParser(description="Deep-learning transfer experiment")
    parser.add_argument("--offline", action="store_true",
                        help="use cached real data instead of downloading")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    print("=" * 78)
    print("DEEP-LEARNING TRANSFER EXPERIMENT")
    print("PyTorch NN  +  TensorFlow hybrid  trained on ALL synthetic data")
    print("then tested on real La Liga 2025/26 and unseen Premier League 2025/26")
    print("=" * 78)

    # ---- train on all synthetic data
    print("\n[1/3] Training on ALL synthetic data (1,200 matches, seed 42)...")
    X, y, P, poisson, df_syn = synthetic_features()

    nn_model = NNFootballPredictor(epochs=args.epochs)
    print("  Training PyTorch NN...")
    nn_model.train(X, y)

    tf_model = TFHybridPredictor(epochs=args.epochs)
    print("  Training TF hybrid (NN + PoissonElo probabilities)...")
    tf_model.train(X, P, y, verbose=False)

    gb = MLFootballPredictor(model_type="gradient_boosting")
    print("  Training sklearn Gradient Boosting (pipeline reference)...")
    feat_syn = NNFootballPredictor.build_features(df_syn)
    gb.train(feat_syn, verbose=False)

    print("  Training sklearn baselines (Logistic / Ridge / Random Forest)...")
    lr = fit_sklearn_baseline(LogisticRegression(max_iter=2000, C=0.5), X, y)
    ridge = fit_sklearn_baseline(RidgeClassifier(alpha=1.0), X, y)
    rf = fit_sklearn_baseline(
        RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                               random_state=42), X, y)

    # in-sample sanity (labeled as such - the models saw this data)
    is_acc = float(np.mean(np.argmax(nn_model.predict_proba_matrix(X), axis=1) == y))
    print(f"\n  [info] In-sample accuracy on the training data: NN {is_acc:.1%} "
          f"(expected - models memorised the synthetic world; the real-league "
          f"results below are the honest out-of-distribution test)")

    models = {"nn": nn_model, "tf": tf_model, "gb": gb, "lr": lr,
              "ridge": ridge, "rf": rf}
    results = []

    # ---- cross-league: La Liga
    print("\n[2/3] CROSS-LEAGUE TEST - real La Liga 2025/26")
    run_league(models, "SP1", args.offline, results, poisson)

    # ---- out-of-sample: Premier League
    print("\n[3/3] OUT-OF-SAMPLE TEST - real Premier League 2025/26 (unseen matches)")
    run_league(models, "E0", args.offline, results, poisson)

    # ---- save
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "transfer_results.csv", index=False)
    print(f"\n[OK] Saved {RESULTS_DIR / 'transfer_results.csv'}")

    _plot_transfer(res_df)
    _write_summary_doc(res_df)


def _plot_transfer(res_df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["Base rate (most common)", "Market (bookmaker odds)",
             "PyTorch NN", "TF hybrid (NN + PoissonElo)",
             "sklearn Gradient Boosting", "sklearn Logistic Regression",
             "sklearn Ridge classifier", "sklearn Random Forest",
             "PyTorch NN - cold start (no team info)"]
    for league in res_df["league"].unique():
        sub = res_df[res_df["league"] == league].set_index("method")
        sub = sub.reindex([m for m in order if m in sub.index])
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = ["#9aa0a6", "#2E86AB", "#A23B72", "#F18F01", "#06D6A0",
                  "#5b8def", "#7d5ba6", "#4db6ac", "#dddddd"]
        ax.barh(sub.index, sub["accuracy"] * 100, color=colors[: len(sub)])
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"Transfer accuracy - {league} 2025/26 (models trained on synthetic data)")
        ax.set_xlim(0, max(sub["accuracy"].max() * 100 + 5, 60))
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(row["accuracy"] * 100 + 0.5, i, f"{row['accuracy']:.1%}",
                    va="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"transfer_{league.replace(' ', '_').lower()}.png",
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved transfer_{league.replace(' ', '_').lower()}.png")


def _write_summary_doc(res_df: pd.DataFrame):
    lines = [
        "# Deep-Learning Transfer Experiment",
        "",
        "Two deep models - a **PyTorch MLP** and a **TensorFlow hybrid** (an MLP",
        "fused with the PoissonElo model's probability outputs) - were trained on",
        "**all 1,200 synthetic matches** (seed 42) and then evaluated on *real*",
        "match data they had never seen:",
        "",
        "| Test set | Question |",
        "|----------|----------|",
        "| La Liga 2025/26 (SP1) | Cross-league: does the learned feature→probability mapping transfer to a different league? |",
        "| Premier League 2025/26 (E0) | Out-of-sample: does it work on real matches in the same league the synthetic data mimics? |",
        "",
        "Features for the real leagues are computed from the **previous** real season",
        "(2024/25): Elo ratings and shifted rolling form. A cold-start row (no team",
        "information at all) shows the null result.",
        "",
        "## Results",
        "",
        "```",
        res_df.to_string(index=False),
        "```",
        "",
        "*(Saved by `scripts/04_deep_learning_transfer.py`; full numbers in",
        "`backtests/results/transfer_results.csv`.)*",
        "",
        "## What this demonstrates",
        "",
        "- If the models beat the base rate but trail the market, the synthetic-trained",
        "  mapping transfers *partially*: it learned something real about football, but",
        "  the real bookmaker remains the strongest predictor.",
        "- The cold-start row quantifies how much of the accuracy comes from team",
        "  information vs prior probabilities alone.",
        "- The conclusion for real deployment: retrain on real data (see",
        "  `scripts/01_data_ingestion.py`); the synthetic world validates the",
        "  methodology, not the absolute numbers.",
    ]
    doc = PROJECT_ROOT / "docs" / "04_deep_learning_transfer.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {doc}")


if __name__ == "__main__":
    main()
