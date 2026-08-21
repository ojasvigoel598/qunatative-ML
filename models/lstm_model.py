#!/usr/bin/env python3
"""
STATEFUL SEQUENCE MODEL (LSTM / GRU) for match-outcome prediction.

Motivation
----------
A match is not an isolated event: a team arrives with a *state* built from its
recent matches (form, fatigue, momentum, tactical evolution).  Feed-forward
models (GB, MLP) see one flat feature vector.  This model instead encodes the
rolling sequence of each team's last K matches with an LSTM/GRU — the hidden
state at the end of the sequence IS the team's learned state vector — and a
fused head turns (state_home, state_away, static) into P(H/D/A).

Leakage-free design
-------------------
* Team histories are maintained ONLINE: the sequence for match *t* contains
  only matches strictly before *t*.
* Per-step features are historical FACTS (goals, result, shots, cards) plus a
  home/away flag — all known after the match, none from the future.
* Static features (Elo difference, bookmaker implied probabilities) are known
  before kick-off.
* A chronological walk over the test set updates histories after each match,
  exactly as a deployed system would.

Both LSTM and GRU cells are supported (--cell), and the rich feature set
(shots, corners, cards, odds) degrades gracefully to goals-only when those
columns are absent (e.g. a different sport).

Usage
-----
    from models.lstm_model import StatefulSequenceModel
    m = StatefulSequenceModel(rich=True, cell="lstm")
    m.train(train_df, valid_df)          # chronological validation split
    probs = m.predict(home, away)        # uses current online state
    m.observe(home, away, hg, ag, result)# reveal -> update state
"""

from __future__ import annotations

import math
import sys
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.adaptive_model import CLASS_MAP, ELO_BASE, OnlineState  # noqa: E402

SEQ_LEN = 8
HIDDEN = 32
LAYERS = 1
STATIC_DIM = 5          # elo_diff + implied H/D/A (from odds) + form diff


def _step_features(row, rich: bool) -> list:
    """Encode one HISTORICAL match as a team-state step (facts only)."""
    f = [
        float(row["home_goals"]),   # goals scored by the team we track if home
        float(row["away_goals"]),
        1.0 if row["result"] == "H" else 0.0,
        1.0 if row["result"] == "D" else 0.0,
        1.0 if row["result"] == "A" else 0.0,
    ]
    if rich:
        for c in ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY"]:
            v = row.get(c, None)
            f.append(float(v) if pd.notna(v) else 0.0)
        # implied probabilities from the pre-match B365 odds at that match
        # (renamed to odds_home/odds_draw/odds_away by the rich data loader;
        # the old code read the raw B365H column names, which no longer exist
        # in the rich schema, so the market signal silently collapsed to 1/3)
        for c in ["odds_home", "odds_draw", "odds_away"]:
            v = row.get(c, None)
            f.append(1.0 / float(v) if pd.notna(v) and float(v) > 1 else 1 / 3.0)
    return f


class _TeamHistory:
    """Online per-team rolling history (append-only, chronological)."""

    def __init__(self, rich: bool):
        self.rich = rich
        self.store: dict = defaultdict(lambda: deque(maxlen=SEQ_LEN))

    def append(self, team: str, row):
        self.store[team].append(_step_features(row, self.rich))

    def seq(self, team: str, feat_dim: int) -> np.ndarray:
        dq = self.store[team]
        seq = np.zeros((SEQ_LEN, feat_dim), dtype=np.float32)
        off = SEQ_LEN - len(dq)
        for i, f in enumerate(dq):
            seq[off + i, :len(f)] = f
        return seq


class _StateCell(nn.Module):
    def __init__(self, feat_dim, hidden, layers, cell):
        super().__init__()
        cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = cls(feat_dim, hidden, layers, batch_first=True)
        self.hidden = hidden

    def forward(self, x):
        # x: (B, L, F) -> use last hidden state
        out, _ = self.rnn(x)
        return out[:, -1, :]


class _FusedNet(nn.Module):
    """Shared RNN over (home_seq, away_seq) fused with a static vector."""

    def __init__(self, feat_dim, hidden, layers, cell, static_dim,
                 activation: str = "elu", dropout: float = 0.2):
        super().__init__()
        self.rnn = _StateCell(feat_dim, hidden, layers, cell)
        
        # Choose activation (research-backed: ELU for small data)
        ACTIVATIONS = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
            "elu": nn.ELU(alpha=1.0),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "mish": nn.Mish(),
        }
        act_fn = ACTIVATIONS.get(activation, nn.ELU())
        
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + static_dim, 64),
            nn.LayerNorm(64),  # LayerNorm instead of BatchNorm for sequence models
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        xh, xa, xs = x
        h = self.rnn(xh)
        a = self.rnn(xa)
        return self.head(torch.cat([h, a, xs], dim=1))


class StatefulSequenceModel:
    """LSTM/GRU over rolling team sequences + static fusion head."""

    def __init__(self, rich: bool = True, cell: str = "lstm", hidden: int = HIDDEN,
                 layers: int = LAYERS, seq_len: int = SEQ_LEN, seed: int = 42,
                 epochs: int = 60, lr: float = 2e-3,
                 activation: str = "elu", dropout: float = 0.2):
        self.rich = rich
        self.cell = cell
        self.hidden = hidden
        self.layers = layers
        self.seq_len = seq_len
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.activation = activation
        self.dropout = dropout
        self.net = None
        self.scaler = None
        self.team_state = _TeamHistory(rich)
        self.elo_state = OnlineState()
        self.static_mean = None
        self.static_std = None
        self.feat_dim = 5 + (8 + 3 if rich else 0)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------ data prep
    def _static_vec(self, home: str, away: str, odds: dict = None) -> np.ndarray:
        f = self.elo_state.features(home, away)   # elo_diff + goal avgs + pts
        elo_diff = f["elo_diff"]
        if odds and all(pd.notna(odds.get(k)) for k in ("home_win", "draw", "away_win")):
            impl = np.array([1 / odds["home_win"], 1 / odds["draw"], 1 / odds["away_win"]])
            impl = impl / impl.sum()
        else:
            impl = np.array([1 / 3.0] * 3)
        form_diff = f["home_pts_5"] - f["away_pts_5"]
        return np.array([elo_diff / 400.0, *impl, form_diff], dtype=np.float32)

    def _build_tensor(self, home: str, away: str, odds: dict = None):
        seq_h = torch.tensor(self.team_state.seq(home, self.feat_dim)).unsqueeze(0)
        seq_a = torch.tensor(self.team_state.seq(away, self.feat_dim)).unsqueeze(0)
        stat = torch.tensor(self._static_vec(home, away, odds)).unsqueeze(0)
        return seq_h, seq_a, stat

    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame):
        """Train on the chronological train split, early-stop on the valid split.

        Histories are built online over train+valid in order, so no sample
        ever sees its own future.  The model state is left advanced exactly up
        to the last training match (predictions on the test walk continue it).
        """
        # ---- build sequences for train & valid (online, in chronological order)
        Xh, Xa, Xs, y = [], [], [], []
        for df in (train_df, valid_df):
            for _, r in df.iterrows():
                odds = {"home_win": r.get("odds_home"), "draw": r.get("odds_draw"),
                        "away_win": r.get("odds_away")}
                sh, sa, st = self._build_tensor(r["home_team"], r["away_team"], odds)
                Xh.append(sh); Xa.append(sa); Xs.append(st)
                y.append(CLASS_MAP[r["result"]])
                # reveal -> update histories (the NEXT sample sees this match)
                self.team_state.append(r["home_team"], r)
                self.team_state.append(r["away_team"], r)
                self.elo_state.update(r["home_team"], r["away_team"],
                                      float(r["home_goals"]), float(r["away_goals"]),
                                      r["result"])

        Xh = torch.cat(Xh); Xa = torch.cat(Xa); Xs = torch.cat(Xs)
        y = torch.tensor(y, dtype=torch.long)
        n_train = len(train_df)

        # ---- standardize static features (fit on train only)
        tr_s = Xs[:n_train]
        self.static_mean = tr_s.mean(0, keepdim=True)
        self.static_std = tr_s.std(0, keepdim=True).clamp_min(1e-4)
        Xs = (Xs - self.static_mean) / self.static_std

        # ---- net: shared RNN on both team sequences + fused static head
        self.net = _FusedNet(self.feat_dim, self.hidden, self.layers,
                             self.cell, STATIC_DIM,
                             activation=self.activation, dropout=self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        best_ll, best_state, patience = float("inf"), None, 0

        Xv_h, Xv_a, Xv_s = Xh[n_train:], Xa[n_train:], Xs[n_train:]
        yv = y[n_train:]
        Xh_t, Xa_t, Xs_t = Xh[:n_train], Xa[:n_train], Xs[:n_train]
        y_t = y[:n_train]

        for ep in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            logits = self.net((Xh_t, Xa_t, Xs_t))
            loss = loss_fn(logits, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()

            if len(Xv_h):
                self.net.eval()
                with torch.no_grad():
                    lv = loss_fn(self.net((Xv_h, Xv_a, Xv_s)), yv).item()
                if lv < best_ll:
                    best_ll = lv
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 8:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return {"valid_log_loss": round(best_ll, 4) if best_ll < float("inf") else None,
                "epochs_run": ep + 1}

    # ------------------------------------------------------------ inference
    def predict(self, home: str, away: str, odds: dict = None) -> dict:
        if self.net is None:
            raise ValueError("train() first")
        sh, sa, st = self._build_tensor(home, away, odds)
        st = (st - self.static_mean) / self.static_std
        with torch.no_grad():
            logits = self.net((sh, sa, st))
            p = torch.softmax(logits, dim=1)[0].numpy()
        return {"away_win": round(float(p[0]), 4),
                "draw": round(float(p[1]), 4),
                "home_win": round(float(p[2]), 4)}

    def observe(self, home: str, away: str, hg: float, ag: float, result: str, row=None):
        self.team_state.append(home, row)
        self.team_state.append(away, row)
        self.elo_state.update(home, away, float(hg), float(ag), result)

    def probe(self) -> dict:
        return {"cell": self.cell, "rich": self.rich,
                "feat_dim": self.feat_dim, "hidden": self.hidden}


if __name__ == "__main__":
    # smoke test: tiny synthetic league, LSTM vs GRU
    rng = np.random.default_rng(0)
    n = 500
    teams = [f"T{i}" for i in range(12)]
    df = pd.DataFrame({
        "home_team": rng.choice(teams, n), "away_team": rng.choice(teams, n),
        "home_goals": rng.poisson(1.6, n), "away_goals": rng.poisson(1.2, n),
        "HS": rng.poisson(12, n), "AS": rng.poisson(10, n),
    })
    df = df[df["home_team"] != df["away_team"]].reset_index(drop=True)
    df["result"] = np.where(df["home_goals"] > df["away_goals"], "H",
                            np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    df["odds_home"], df["odds_draw"], df["odds_away"] = 2.5, 3.2, 3.0

    for cell in ("lstm", "gru"):
        m = StatefulSequenceModel(rich=True, cell=cell, epochs=25)
        m.train(df.iloc[:380], df.iloc[380:430])
        accs = []
        for _, r in df.iloc[430:].iterrows():
            p = m.predict(r["home_team"], r["away_team"])
            accs.append(CLASS_MAP[r["result"]] == int(np.argmax(
                [p["away_win"], p["draw"], p["home_win"]])))
            m.observe(r["home_team"], r["away_team"], r["home_goals"], r["away_goals"],
                      r["result"], row=r)
        print(f"[OK] {cell}: acc={np.mean(accs):.3f}")
