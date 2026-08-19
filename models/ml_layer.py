#!/usr/bin/env python3
"""
ML Layer: Gradient Boosting / Random Forest classifier for match outcomes.

The ML layer consumes the Elo features produced by the PoissonEloModel plus
short-term team form (rolling average goals), and predicts P(home win),
P(draw), P(away win).  Its probabilities are averaged with the PoissonElo
probabilities to form the hybrid prediction used in the backtest.

Correctness notes
-----------------
* Rolling form features use ``.shift(1)`` so a match's own goals are never
  used as a feature for that same match (no target leakage).
* At prediction time the model uses the *actual* Elo ratings of the two teams
  and the per-team form stored at the end of training, instead of constant
  placeholders.  The original code fed constant features to ``predict_proba``,
  which made the model unable to distinguish teams.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

BASE_HOME_GOALS = 1.6
BASE_AWAY_GOALS = 1.3
BASE_ELO = 1500.0


class MLFootballPredictor:
    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        if model_type == "lightgbm":
            if not HAS_LIGHTGBM:
                raise ImportError(
                    "lightgbm is not installed.  "
                    "Install with: pip install lightgbm"
                )
            base = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )
        elif model_type == "gradient_boosting":
            base = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, subsample=0.8, random_state=42,
            )
        else:
            base = RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42,
            )
        # Probability calibration is essential: raw tree ensembles give extreme
        # probabilities, and edge = p * odds - 1 is extremely sensitive to
        # miscalibration.  CalibratedClassifierCV fits an internal 3-fold
        # sigmoid calibration on the training data (no test leakage).
        # Calibration folds must respect time order.  Random folds can make
        # the diagnostics look better by allowing later matches to calibrate
        # earlier ones; the final model is evaluated only after the training
        # window, so use temporal folds throughout.
        self.model = CalibratedClassifierCV(
            base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))
        self.is_trained = False
        self.validation_protocol = "chronological_holdout_80_20"
        self.feature_cols = [
            "home_elo", "away_elo", "home_goals_avg", "away_goals_avg",
        ]
        # Per-team form, stored at the end of training and used for
        # out-of-sample predictions (avoids constant/placeholder features).
        self.team_home_form: pd.Series = pd.Series(dtype=float)
        self.team_away_form: pd.Series = pd.Series(dtype=float)

    # ------------------------------------------------------- Features
    def prepare_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Add rolling (shifted) form features and Elo columns if missing."""
        df = df.copy()

        # Shifted rolling means: the feature for match i uses matches i-5..i-1,
        # never the current match -> no leakage.
        df["home_goals_avg"] = (
            df.groupby("home_team")["home_goals"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )
        df["away_goals_avg"] = (
            df.groupby("away_team")["away_goals"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

        if "home_elo" not in df.columns:
            df["home_elo"] = BASE_ELO
            df["away_elo"] = BASE_ELO

        return df

    def _store_team_form(self, df: pd.DataFrame):
        """Remember each team's latest form so future matches can be scored."""
        home = df.dropna(subset=["home_goals_avg"]).groupby("home_team")["home_goals_avg"].last()
        away = df.dropna(subset=["away_goals_avg"]).groupby("away_team")["away_goals_avg"].last()
        self.team_home_form = home
        self.team_away_form = away

    # ------------------------------------------------------- Train
    def train(self, historical_df: pd.DataFrame, verbose: bool = True):
        """Train on a dataframe that already contains home_elo/away_elo.

        Pass the output of ``PoissonEloModel.prepare_features`` (i.e. the Elo
        features computed on the *training* split only).
        """
        print("Training ML Layer (Gradient Boosting / Random Forest)...")
        df = self.prepare_features(historical_df)

        target_map = {"H": 2, "D": 1, "A": 0}
        y = df["result"].map(target_map)

        X = df[self.feature_cols].copy()
        # A team with no prior home games gets the league-average baseline.
        X["home_elo"] = X["home_elo"].fillna(BASE_ELO)
        X["away_elo"] = X["away_elo"].fillna(BASE_ELO)
        X["home_goals_avg"] = X["home_goals_avg"].fillna(BASE_HOME_GOALS)
        X["away_goals_avg"] = X["away_goals_avg"].fillna(BASE_AWAY_GOALS)

        # Drop any rows with a missing target (defensive).
        mask = y.notna()
        X, y = X[mask], y[mask].astype(int)

        # Keep the model diagnostic temporal.  The final backtest still
        # evaluates on a later, untouched split; this holdout only reports
        # whether training is behaving sensibly without random reordering.
        split_at = max(1, int(len(X) * 0.8))
        X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
        y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
        if y_test.nunique() < 2:
            # Tiny synthetic fixtures can have a degenerate tail.  Keep the
            # temporal fit valid and report the diagnostic on the full window.
            X_test, y_test = X_train, y_train

        self.model.fit(X_train, y_train)
        self.is_trained = True
        # Store end-of-training state for future out-of-sample fixtures.  The
        # diagnostic above is scored from explicit temporal feature rows, so it
        # does not use these stored values.
        self._store_team_form(df)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        proba = self.model.predict_proba(X_test)
        ll = log_loss(y_test, proba, labels=[0, 1, 2])

        if verbose:
            print(f"  ML test accuracy: {acc:.3f} | log-loss: {ll:.3f} (3-class baseline log-loss {np.log(3):.3f})")
            if hasattr(self.model, "feature_importances_"):
                imp = pd.Series(self.model.feature_importances_, index=self.feature_cols)
                print("  Top features:", imp.sort_values(ascending=False).head(3).round(4).to_dict())

        return {"accuracy": float(acc), "log_loss": float(ll)}

    # ------------------------------------------------------- Predict
    def predict_proba_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for a full feature matrix (n, 4).

        Columns must match ``feature_cols``; used by the transfer experiment.
        Returns an (n, 3) matrix ordered [away_win, draw, home_win].
        """
        if not self.is_trained:
            raise ValueError("ML model not trained")
        return self.model.predict_proba(X[self.feature_cols].fillna(0))

    def predict_proba(self, home_team: str, away_team: str,
                      home_elo: float = BASE_ELO, away_elo: float = BASE_ELO) -> dict:
        """Predict outcome probabilities for one fixture.

        Args:
            home_elo / away_elo: current Elo ratings (from the PoissonEloModel).
        """
        if not self.is_trained:
            raise ValueError("ML model not trained")

        home_form = float(self.team_home_form.get(home_team, BASE_HOME_GOALS))
        away_form = float(self.team_away_form.get(away_team, BASE_AWAY_GOALS))

        features = pd.DataFrame([{
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_goals_avg": home_form,
            "away_goals_avg": away_form,
        }])

        proba = self.model.predict_proba(features)[0]
        return {
            "away_win": round(float(proba[0]), 4),
            "draw": round(float(proba[1]), 4),
            "home_win": round(float(proba[2]), 4),
        }


if __name__ == "__main__":
    from poisson_elo_model import PoissonEloModel  # noqa: E402

    rng = np.random.default_rng(0)
    teams = ["Arsenal", "Man City", "Liverpool", "Chelsea"]
    sample = pd.DataFrame({
        "home_team": rng.choice(teams, 400),
        "away_team": rng.choice(teams, 400),
        "home_goals": rng.poisson(1.7, 400),
        "away_goals": rng.poisson(1.3, 400),
    })
    sample["result"] = np.where(sample["home_goals"] > sample["away_goals"], "H",
                                np.where(sample["home_goals"] < sample["away_goals"], "A", "D"))
    p = PoissonEloModel()
    feat = p.prepare_features(sample)
    ml = MLFootballPredictor()
    ml.train(feat)
    probs = ml.predict_proba("Arsenal", "Chelsea", home_elo=p.get_team_elo("Arsenal"),
                             away_elo=p.get_team_elo("Chelsea"))
    print("Arsenal vs Chelsea:", probs)
    assert abs(sum(probs.values()) - 1.0) < 0.01
    print("[OK] MLFootballPredictor self-test passed.")
