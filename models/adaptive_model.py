#!/usr/bin/env python3
"""
ADAPTIVE MATCH PREDICTOR — a baseline ML model with an online-adapting layer.

Why adaptive?
-------------
A model trained on one league (or sport) reflects that league's statistics:
home-advantage size, scoring rate, draw frequency.  When you point it at a
DIFFERENT league or a DIFFERENT sport, those statistics shift and a frozen
model degrades.  Real matches arrive one at a time, so we can adapt:

  1. BASE LAYER   — PoissonElo trained on the initial data (the project core).
  2. ONLINE STATE — Elo ratings and rolling form updated AFTER every match, so
                    every prediction uses only information known before
                    kick-off (no leakage).
  3. FEATURES     — league-agnostic: Elo difference, rolling goals, form
                    points.  NO team identity, NO league constants, so the same
                    feature vector works in Serie A, the Premier League, La
                    Liga, or a different sport.
  4. ML LAYER     — Gradient Boosting over those features, calibrated.
  5. ADAPTATION   — two triggers refit the ML layer on the most recent window:
                      * scheduled: every `refit_every` matches;
                      * drift:     rolling Brier on recent predictions degrades
                                   vs the baseline by more than `drift_tol`.
                    A `static` mode (fit once, never refit) is provided as the
                    control to measure exactly what adaptation buys.

Usage
-----
    from models.adaptive_model import AdaptiveMatchPredictor
    m = AdaptiveMatchPredictor()
    m.train(initial_df)                  # fit base + ML on initial data
    probs = m.predict("Inter", "Juventus")
    m.observe(real_match_row)            # reveal result -> adapt
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# allow standalone execution (`python models/adaptive_model.py`)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ELO_K = 20.0
ELO_BASE = 1500.0
CLASS_MAP = {"A": 0, "D": 1, "H": 2}
OUTCOME_NAMES = ["away_win", "draw", "home_win"]

FEATURE_COLS = [
    "elo_diff",          # home_elo - away_elo
    "home_goals_avg",    # last-5 home goals scored
    "away_goals_avg",    # last-5 away goals conceded
    "home_pts_5",        # points per last-5 home matches
    "away_pts_5",        # points per last-5 away matches
]


class OnlineState:
    """Running Elo + rolling form for ANY team (league/sport agnostic)."""

    def __init__(self):
        self.elo = defaultdict(lambda: ELO_BASE)
        self.home_goals: dict = defaultdict(list)   # goals scored at home
        self.away_goals: dict = defaultdict(list)   # goals conceded away
        self.home_pts: dict = defaultdict(list)     # points won at home
        self.away_pts: dict = defaultdict(list)     # points won away

    def features(self, home: str, away: str) -> dict:
        elo_diff = self.elo[home] - self.elo[away]
        return {
            "elo_diff": float(elo_diff),
            "home_goals_avg": self._avg(self.home_goals[home]),
            "away_goals_avg": self._avg(self.away_goals[away]),
            "home_pts_5": self._avg(self.home_pts[home]),
            "away_pts_5": self._avg(self.away_pts[away]),
        }

    def update(self, home: str, away: str, hg: float, ag: float, result: str):
        home_pts = 3.0 if result == "H" else (1.0 if result == "D" else 0.0)
        away_pts = 3.0 if result == "A" else (1.0 if result == "D" else 0.0)

        exp_home = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home]) / 400))
        actual = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        self.elo[home] += ELO_K * (actual - exp_home)
        self.elo[away] += ELO_K * ((1 - actual) - (1 - exp_home))

        self.home_goals[home].append(float(hg))
        self.away_goals[away].append(float(ag))
        self.home_pts[home].append(home_pts)
        self.away_pts[away].append(away_pts)

    @staticmethod
    def _avg(xs, default: float = 1.4):
        return float(np.mean(xs[-5:])) if xs else default


def brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    return float(np.mean(np.sum((probs - np.eye(3)[y_true]) ** 2, axis=1)))


class AdaptiveMatchPredictor:
    """Baseline PoissonElo + GB with online adaptation (or static control)."""

    def __init__(self, window: int = 240, refit_every: int = 60,
                 drift_tol: float = 0.02, min_refit: int = 60,
                 static: bool = False, seed: int = 42):
        """
        Args:
            window:      rolling window of recent matches used for refits.
            refit_every: scheduled refit cadence (matches).
            drift_tol:   refit when rolling Brier degrades by more than this
                         versus the best rolling Brier seen so far.
            min_refit:   minimum matches before the first refit (cold start).
            static:      control mode — fit once, NEVER refit the ML layer
                         (Elo/form still update online).
        """
        self.window = window
        self.refit_every = refit_every
        self.drift_tol = drift_tol
        self.min_refit = min_refit
        self.static = static
        self.frozen = False   # runtime freeze (final-validation window): stop refits
        self.seed = seed

        from models.poisson_elo_model import PoissonEloModel
        self.poisson = PoissonEloModel(elo_k=ELO_K)
        self.state = OnlineState()
        self.ml = None
        self.base_rates = np.array([0.27, 0.28, 0.45])  # league-agnostic prior
        self.history = deque(maxlen=max(window * 4, 240))  # (features, y)
        self.seen = 0
        self.best_rolling_brier = 1.0
        self.refits = 0
        self._buffer_x, self._buffer_y = [], []
        self._recent_brier = deque(maxlen=40)   # observed Brier per match
        self._drift_armed = 0                     # matches since last refit

    # ------------------------------------------------------------- training
    def train(self, df: pd.DataFrame):
        """Fit the PoissonElo base and the ML layer on initial data."""
        self.poisson.train(df)
        base = df["result"].value_counts(normalize=True)
        self.base_rates = np.array([base.get("A", 0.27), base.get("D", 0.28),
                                    base.get("H", 0.45)])
        # league-agnostic prior probs for matches with no ML yet
        self.prior = self.base_rates / self.base_rates.sum()

        X = self._build_features(df)
        y = df["result"].map(CLASS_MAP).to_numpy()
        self.ml = self._fit_ml(X, y)
        # pre-warm the online state over the whole training set (online, no leak)
        for _, r in df.iterrows():
            self.state.update(r["home_team"], r["away_team"],
                              float(r["home_goals"]), float(r["away_goals"]),
                              r["result"])
            self._buffer_x.append(self._feat_row(r["home_team"], r["away_team"],
                                                 r["result"]))
            self._buffer_y.append(CLASS_MAP[r["result"]])
        self._buffer_x = self._buffer_x[-self.window * 4:]
        self._buffer_y = self._buffer_y[-self.window * 4:]

    def _feat_row(self, home, away, _result=None):
        f = self.state.features(home, away)
        return [f[c] for c in FEATURE_COLS]

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            f = self.state.features(r["home_team"], r["away_team"])
            rows.append([f[c] for c in FEATURE_COLS])
            self.state.update(r["home_team"], r["away_team"],
                              float(r["home_goals"]), float(r["away_goals"]),
                              r["result"])
        # reset online state: features were built online but training must not
        # leave the state advanced (predictions will advance it again)
        self.state = OnlineState()
        return pd.DataFrame(rows, columns=FEATURE_COLS)

    def _fit_ml(self, X: pd.DataFrame, y: np.ndarray):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import GradientBoostingClassifier
        # Cheaper model for online refits: fewer estimators and a 2-fold
        # calibration CV keep the rolling refits fast enough for the $1M
        # Monte-Carlo loop (25 trials x ~20 refits each).
        gb = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.8, random_state=self.seed)
        if len(np.unique(y)) < 2 or X.shape[0] < 30:
            return None
        try:
            return CalibratedClassifierCV(gb, method="sigmoid", cv=2).fit(X, y)
        except Exception:
            return gb.fit(X, y)

    # ------------------------------------------------------------ prediction
    def predict(self, home: str, away: str) -> dict:
        p = self.poisson.predict(home, away)
        poisson_vec = np.array([p["away_win"], p["draw"], p["home_win"]])
        if self.ml is not None:
            f = self.state.features(home, away)
            X = pd.DataFrame([[f[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
            raw = self.ml.predict_proba(X)[0]
            # ML may have seen only 2 classes (e.g. a no-draw sport): map its
            # classes back into the fixed 3-class vector.
            ml_vec = np.zeros(3)
            for cls, p in zip(self.ml.classes_, raw):
                ml_vec[int(cls)] = p
            if ml_vec.sum() <= 0:
                ml_vec = self.prior
            vec = 0.6 * poisson_vec + 0.4 * ml_vec
        else:
            vec = poisson_vec
        vec = np.clip(vec / vec.sum(), 1e-6, 1.0)
        vec = vec / vec.sum()
        return {"away_win": round(float(vec[0]), 4),
                "draw": round(float(vec[1]), 4),
                "home_win": round(float(vec[2]), 4)}

    # ------------------------------------------------------------- adaptation
    def observe(self, home: str, away: str, hg: float, ag: float,
                result: str, prob_vec: np.ndarray = None):
        """Reveal a played match: update online state + maybe refit.

        Call AFTER using `predict` for the match.  `prob_vec` is the model's
        prediction for this match (for drift monitoring).
        """
        self.state.update(home, away, float(hg), float(ag), result)
        self._buffer_x.append(self._feat_row(home, away))
        self._buffer_y.append(CLASS_MAP[result])
        self._buffer_x = self._buffer_x[-self.window * 4:]
        self._buffer_y = self._buffer_y[-self.window * 4:]
        self.seen += 1

        if prob_vec is not None:
            rb = brier(np.array([CLASS_MAP[result]]),
                       np.array([prob_vec]))
            self._recent_brier.append(rb)
            self.best_rolling_brier = min(self.best_rolling_brier,
                                          float(np.mean(self._recent_brier)))

        if self.static or self.frozen or self.seen < self.min_refit:
            return

        drift = self._drift_detected()
        due = (self.seen % self.refit_every) == 0
        if drift or due:
            self._refit()
            self._drift_armed = 0
        else:
            self._drift_armed += 1

    def _drift_detected(self) -> bool:
        """Refit when the rolling observed Brier degrades vs its best by tol.

        Uses ONLY predictions actually made before each match (observed Brier),
        so the trigger itself never sees the future.
        """
        if len(self._recent_brier) < 30 or self._drift_armed < 10:
            return False
        recent = float(np.mean(self._recent_brier))
        return recent > self.best_rolling_brier + self.drift_tol

    def _refit(self):
        if len(self._buffer_y) < 30 or len(set(self._buffer_y)) < 2:
            return
        X = pd.DataFrame(self._buffer_x[-self.window:], columns=FEATURE_COLS)
        y = np.array(self._buffer_y[-self.window:])
        new = self._fit_ml(X, y)
        if new is not None:
            self.ml = new
            self.refits += 1

    def freeze(self):
        """Freeze the ML layer for an untouched evaluation window.

        Elo/form still update online (that is the model's live behaviour), but
        no further ML refits are allowed — the model becomes static for the
        remainder of the walk.  Used for the strict final-validation period.
        """
        self.frozen = True

    def refit_now(self):
        """Public hook: force an immediate ML refit.

        Used by confidence-aware adaptation layers (e.g.
        `DynamicThinkingLayer`) that detect the model *losing confidence*
        and want to re-learn from the most recent window right away, in
        addition to the scheduled/drift triggers.
        """
        self._refit()

    # ---------------------------------------------------------------- utils
    def probe(self) -> dict:
        return {"seen": self.seen, "refits": self.refits,
                "static": self.static, "teams": len(self.state.elo)}


if __name__ == "__main__":
    # quick smoke test on a tiny synthetic league
    rng = np.random.default_rng(0)
    n = 400
    teams = [f"T{i}" for i in range(12)]
    df = pd.DataFrame({
        "home_team": rng.choice(teams, n), "away_team": rng.choice(teams, n),
        "home_goals": rng.poisson(1.6, n), "away_goals": rng.poisson(1.2, n),
    })
    df = df[df["home_team"] != df["away_team"]].reset_index(drop=True)
    df["result"] = np.where(df["home_goals"] > df["away_goals"], "H",
                            np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    m = AdaptiveMatchPredictor(static=False)
    m.train(df.iloc[:240])
    accs = []
    for _, r in df.iloc[240:].iterrows():
        p = m.predict(r["home_team"], r["away_team"])
        accs.append(CLASS_MAP[r["result"]] == int(np.argmax(
            [p["away_win"], p["draw"], p["home_win"]])))
        m.observe(r["home_team"], r["away_team"], r["home_goals"], r["away_goals"],
                  r["result"])
    print(f"[OK] adaptive smoke: acc={np.mean(accs):.3f} refits={m.refits}")
