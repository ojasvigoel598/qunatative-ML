#!/usr/bin/env python3
"""
TensorFlow hybrid model: an MLP that fuses the engineered features with the
Poisson + Elo statistical model's probability outputs.

The "hybrid" in the name is deliberate: the network input is the concatenation
of

    4 engineered features (home_elo, away_elo, home_form, away_form)
  + 3 PoissonElo probabilities (P_home, P_draw, P_away)

so the deep net learns how to *adjust* the statistical model's output from the
feature evidence.  This is a genuine statistical + deep-learning hybrid, and
it shares its API with the other model layers:

    from models.tf_hybrid import TFHybridPredictor
    hybrid = TFHybridPredictor()
    hybrid.train(features_df, poisson_model, y)
    probs = hybrid.predict_proba(home_elo, away_elo, home_form, away_form,
                                 poisson_probs)

Notes
-----
* Requires TensorFlow.  On Python 3.14 the stable `tensorflow` package has no
  wheels yet; `tf-nightly` (>= 2.22.0.dev) works, see requirements-deep.txt.
* Determinism: global seeds set; CPU-only run by default.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

CLASS_MAP = {"A": 0, "D": 1, "H": 2}
BASE_ELO = 1500.0
BASE_HOME_GOALS = 1.6
BASE_AWAY_GOALS = 1.3


class TFHybridPredictor:
    def __init__(self, hidden: int = 64, epochs: int = 300, lr: float = 1e-3,
                 batch_size: int = 64, activation: str = "elu",
                 use_batch_norm: bool = True, dropout: float = 0.2):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.model = None
        self.mean = None
        self.std = None
        self.is_trained = False
        self.feature_cols = ["home_elo", "away_elo", "home_goals_avg", "away_goals_avg"]
        self.poisson_prob_cols = ["pois_home_win", "pois_draw", "pois_away_win"]

    # ------------------------------------------------------------ Features
    @staticmethod
    def build_poisson_probs(poisson, df: "pd.DataFrame") -> "np.ndarray":
        """PoissonElo probabilities (n, 3) for every row of df."""
        rows = []
        for _, r in df.iterrows():
            p = poisson.predict(r["home_team"], r["away_team"])
            rows.append([p["home_win"], p["draw"], p["away_win"]])
        return np.array(rows, dtype=np.float32)

    def _fit_scaler(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-9

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    # ------------------------------------------------------------ Train
    def train(self, features: np.ndarray, poisson_probs: np.ndarray,
              y: np.ndarray, verbose: bool = True):
        """features: (n, 4) engineered; poisson_probs: (n, 3); y: (n,) classes."""
        X = np.concatenate([np.asarray(features, dtype=np.float32),
                            np.asarray(poisson_probs, dtype=np.float32)], axis=1)
        y = np.asarray(y, dtype=np.int64)
        self._fit_scaler(X)
        Xs = self._scale(X)

        tf.keras.utils.set_random_seed(42)
        inputs = tf.keras.Input(shape=(X.shape[1],))
        
        # Choose activation function
        ACTIVATIONS = {
            "relu": "relu",
            "leaky_relu": "leaky_relu",
            "elu": "elu",
            "gelu": "gelu",
            "swish": "swish",
            "mish": "mish",
        }
        act_fn = ACTIVATIONS.get(self.activation, "elu")
        
        x = tf.keras.layers.Dense(self.hidden, activation=act_fn)(inputs)
        if self.use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(self.hidden, activation=act_fn)(x)
        if self.use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        out = tf.keras.layers.Dense(3, activation="softmax")(x)
        self.model = tf.keras.Model(inputs, out)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model.fit(Xs, y, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=1 if verbose else 0, shuffle=True)
        self.is_trained = True
        if verbose:
            acc = float(self.model.evaluate(self._scale(X), y, verbose=0)[1])
            print(f"  [OK] TF hybrid trained | in-sample accuracy {acc:.3f}")

    # ------------------------------------------------------------ Predict
    def predict_proba_matrix(self, features: np.ndarray,
                             poisson_probs: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        X = np.concatenate([np.asarray(features, dtype=np.float32),
                            np.asarray(poisson_probs, dtype=np.float32)], axis=1)
        return self.model.predict(self._scale(X), verbose=0)

    def predict_proba(self, home_elo: float, away_elo: float,
                      home_form: float, away_form: float,
                      poisson_probs: dict) -> dict:
        X = np.array([[home_elo, away_elo, home_form, away_form]], dtype=np.float32)
        P = np.array([[poisson_probs["home_win"], poisson_probs["draw"],
                       poisson_probs["away_win"]]], dtype=np.float32)
        p = self.predict_proba_matrix(X, P)[0]
        return {"away_win": round(float(p[0]), 4),
                "draw": round(float(p[1]), 4),
                "home_win": round(float(p[2]), 4)}

    def log_loss(self, features: np.ndarray, poisson_probs: np.ndarray,
                 y: np.ndarray) -> float:
        probs = self.predict_proba_matrix(features, poisson_probs)
        eps = 1e-9
        return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1))))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import pipeline
    from models.nn_model import NNFootballPredictor

    df = pipeline.generate_match_data(600, seed=42)
    feat = NNFootballPredictor.build_features(df)
    X = feat[["home_elo", "away_elo", "home_goals_avg", "away_goals_avg"]].fillna(
        {"home_elo": 1500.0, "away_elo": 1500.0,
         "home_goals_avg": 1.6, "away_goals_avg": 1.3}).to_numpy()
    y = df["result"].map(CLASS_MAP).to_numpy()

    poisson = pipeline.PoissonEloModel()
    poisson.train(df)
    P = TFHybridPredictor.build_poisson_probs(poisson, df)

    m = TFHybridPredictor(epochs=100)
    m.train(X, P, y, verbose=False)
    print("Arsenal vs Chelsea:", m.predict_proba(1600, 1400, 1.9, 1.1,
                                                 {"home_win": 0.55, "draw": 0.26, "away_win": 0.19}))
    print("[OK] TFHybridPredictor self-test passed.")
