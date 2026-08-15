#!/usr/bin/env python3
"""
Neural Network layer (PyTorch): an MLP for match-outcome prediction.

This is a self-contained deep-learning counterpart to the pipeline's sklearn
ML layer.  It consumes the SAME engineered features (home_elo, away_elo,
home_goals_avg, away_goals_avg) and predicts P(away win), P(draw),
P(home win) with a small multi-layer perceptron.

API mirrors the other model layers:
    nn_model = NNFootballPredictor()
    nn_model.train(X_train, y_train)
    probs = nn_model.predict_proba(home_elo, away_elo, home_form, away_form)

Notes
-----
* Features are z-scored with statistics fitted on the training data (never on
  the evaluation data).
* Fixed random seed -> reproducible training.
* Designed to be trained on the *whole* synthetic dataset and then evaluated
  on held-out / cross-league data (see scripts/04_deep_learning_transfer.py).
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

# Class index mapping (consistent with the rest of the project)
CLASS_MAP = {"A": 0, "D": 1, "H": 2}

# League-average fallbacks used when a team has no history in the training set
BASE_ELO = 1500.0
BASE_HOME_GOALS = 1.6
BASE_AWAY_GOALS = 1.3


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


class NNFootballPredictor:
    def __init__(self, hidden: int = 64, epochs: int = 300, lr: float = 1e-3,
                 batch_size: int = 64):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.mean = None
        self.std = None
        self.is_trained = False
        self.feature_cols = ["home_elo", "away_elo", "home_goals_avg", "away_goals_avg"]

    # ------------------------------------------------------------ Features
    @staticmethod
    def build_features(df: "pd.DataFrame") -> "pd.DataFrame":
        """Engineered features: Elo columns + shifted rolling form.

        Reuses the exact feature logic of the pipeline's models so the neural
        net sees identical inputs to the sklearn ML layer.
        """
        import pandas as pd
        from models.poisson_elo_model import PoissonEloModel
        from models.ml_layer import MLFootballPredictor

        feat = PoissonEloModel().prepare_features(df)
        feat = MLFootballPredictor().prepare_features(feat)
        return feat

    def _fit_scaler(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-9

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    # ------------------------------------------------------------ Train
    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train on a feature matrix (n, 4) and class labels (A=0, D=1, H=2)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._fit_scaler(X)
        Xs = self._scale(X)

        self.model = _MLP(n_in=X.shape[1], hidden=self.hidden)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(y)
        n = len(Xt)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                out = self.model(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 100 == 0:
                print(f"    epoch {epoch + 1:3d}/{self.epochs}  loss {total_loss / max(n // self.batch_size, 1):.4f}")

        self.is_trained = True
        if verbose:
            probs = self.predict_proba_matrix(X)
            acc = float(np.mean(np.argmax(probs, axis=1) == y))
            print(f"  [OK] PyTorch NN trained | in-sample accuracy {acc:.3f}")

    # ------------------------------------------------------------ Predict
    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return an (n, 3) probability matrix ordered [away, draw, home]."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            out = torch.softmax(self.model(torch.from_numpy(self._scale(X))), dim=1)
        return out.numpy()

    def predict_proba(self, home_elo: float = BASE_ELO, away_elo: float = BASE_ELO,
                      home_form: float = BASE_HOME_GOALS,
                      away_form: float = BASE_AWAY_GOALS) -> dict:
        """Predict for one fixture.  Team-independent for unseen teams."""
        X = np.array([[home_elo, away_elo, home_form, away_form]], dtype=np.float32)
        p = self.predict_proba_matrix(X)[0]
        return {"away_win": round(float(p[0]), 4),
                "draw": round(float(p[1]), 4),
                "home_win": round(float(p[2]), 4)}

    def log_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict_proba_matrix(X)
        eps = 1e-9
        return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1))))


if __name__ == "__main__":
    import pandas as pd
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import pipeline
    df = pipeline.generate_match_data(600, seed=42)
    feat = NNFootballPredictor.build_features(df)
    X = feat[["home_elo", "away_elo", "home_goals_avg", "away_goals_avg"]].fillna(
        {"home_elo": 1500.0, "away_elo": 1500.0,
         "home_goals_avg": 1.6, "away_goals_avg": 1.3}).to_numpy()
    y = df["result"].map(CLASS_MAP).to_numpy()
    m = NNFootballPredictor(epochs=100)
    m.train(X, y)
    print("Arsenal vs Chelsea:", m.predict_proba(1600, 1400, 1.9, 1.1))
    print("[OK] NNFootballPredictor self-test passed.")
