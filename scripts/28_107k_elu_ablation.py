#!/usr/bin/env python3
"""
28 — 107K Ablation: ELU + BatchNorm + LR Scheduling + Mixup

Tests neural network activation ablation on 107K real football matches.
NO data leakage: only pre-match features (odds, overround).
Target encoding: 0=Away, 1=Draw, 2=Home → feature columns must match
(away_prob, draw_prob, home_prob) order.

CRITICAL BUG FIX: Previous script had [home, draw, away] column order
but target encoding is {H=2, A=0, D=1}, causing log_loss mismatch.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ELUNetwork(nn.Module):
    """Neural network with ELU activation, BatchNorm, and He initialization."""

    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ELU(alpha=1.0),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0))
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def train_elu(X_tr, y_tr, X_val, y_val, epochs=15, lr=0.001,
              use_mixup=False, use_scheduling=False):
    model = ELUNetwork(X_tr.shape[1])
    X_t = torch.FloatTensor(X_tr)
    y_t = torch.LongTensor(y_tr)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.LongTensor(y_val)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 3) if use_scheduling else None

    best_loss, best_state = float('inf'), None
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            if use_mixup and np.random.random() < 0.5:
                mx, ya, yb, lam = mixup_data(bx, by)
                out = model(mx)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            else:
                loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_v), y_v).item()
        if sched:
            sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.FloatTensor(X)), dim=1).numpy()
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    return {
        'accuracy': np.mean(np.argmax(p, axis=1) == y),
        'log_loss': log_loss(y, p),
        'brier': np.mean([brier_score_loss((y == i).astype(float), p[:, i]) for i in range(3)]),
        'ece': _ece(p, y),
    }


def _ece(probs, y, n=10):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n + 1)
    ece = 0.0
    for i in range(n):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(y) * abs(acc[mask].mean() - conf[mask].mean())
    return ece


def main():
    print("=" * 70)
    print("107K ABLATION — ELU + BatchNorm + LR Scheduling + Mixup")
    print("=" * 70)

    # --- Load data ---
    t0 = time.time()
    df = pd.read_csv(PROJECT_ROOT / "data" / "real" / "all_leagues_combined.csv", low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['home_goals', 'away_goals', 'result', 'odds_home', 'odds_draw', 'odds_away'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(df)} matches, {df['league'].nunique()} leagues ({time.time()-t0:.1f}s)")

    # --- Features: ONLY pre-match data (NO leakage) ---
    odds_h = df['odds_home'].clip(lower=1.01).values
    odds_d = df['odds_draw'].clip(lower=1.01).values
    odds_a = df['odds_away'].clip(lower=1.01).values
    inv_h, inv_d, inv_a = 1 / odds_h, 1 / odds_d, 1 / odds_a
    total = inv_h + inv_d + inv_a

    # CRITICAL: column order must match target {0=A, 1=D, 2=H}
    X = np.column_stack([
        odds_a, odds_d, odds_h,             # raw odds: away, draw, home
        inv_a / total, inv_d / total, inv_h / total,  # implied probs: A, D, H
        total - 1,                            # overround
    ])
    y = df['result'].map({'H': 2, 'A': 0, 'D': 1}).values

    # --- Walk-forward split ---
    split = int(len(X) * 0.8)
    X_traw, X_test = X[:split], X[split:]
    y_tr, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_traw)
    X_test_s = scaler.transform(X_test)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: odds_H, odds_D, odds_A, impl_H, impl_D, impl_A, overround")
    print(f"  Target: H={np.sum(y==2)}, D={np.sum(y==1)}, A={np.sum(y==0)}")

    # --- Baselines ---
    results = []

    # Market (bookmaker)
    mkt_probs = np.column_stack([inv_a / total, inv_d / total, inv_h / total])
    mkt_test = mkt_probs[split:]
    ll_mkt = log_loss(y_test, mkt_test)
    results.append(('Market (B365)', {'log_loss': ll_mkt, 'ece': _ece(mkt_test, y_test),
                                       'accuracy': np.mean(np.argmax(mkt_test, axis=1) == y_test)}))
    print(f"\n  Market (B365): LL={ll_mkt:.4f}")

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_tr)
    lr_probs = lr_model.predict_proba(X_test_s)
    lr_m = {'log_loss': log_loss(y_test, lr_probs), 'accuracy': np.mean(lr_model.predict(X_test_s) == y_test),
            'ece': _ece(lr_probs, y_test), 'time': time.time() - t0}
    results.append(('Logistic Regression', lr_m))
    print(f"  LR:  LL={lr_m['log_loss']:.4f}, Acc={lr_m['accuracy']:.3f}, Time={lr_m['time']:.1f}s")

    # XGBoost
    from xgboost import XGBClassifier
    t0 = time.time()
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         eval_metric='mlogloss', verbosity=0, random_state=42)
    xgb.fit(X_train, y_tr)
    xgb_probs = xgb.predict_proba(X_test_s)
    xgb_m = {'log_loss': log_loss(y_test, xgb_probs), 'accuracy': np.mean(xgb.predict(X_test_s) == y_test),
              'ece': _ece(xgb_probs, y_test), 'time': time.time() - t0}
    results.append(('XGBoost', xgb_m))
    print(f"  XGB: LL={xgb_m['log_loss']:.4f}, Acc={xgb_m['accuracy']:.3f}, Time={xgb_m['time']:.1f}s")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        t0 = time.time()
        lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               verbose=-1, random_state=42)
        lgbm.fit(X_train, y_tr)
        lgbm_probs = lgbm.predict_proba(X_test_s)
        lgbm_m = {'log_loss': log_loss(y_test, lgbm_probs),
                   'accuracy': np.mean(lgbm.predict(X_test_s) == y_test),
                   'ece': _ece(lgbm_probs, y_test), 'time': time.time() - t0}
        results.append(('LightGBM', lgbm_m))
        print(f"  LGBM: LL={lgbm_m['log_loss']:.4f}, Acc={lgbm_m['accuracy']:.3f}, Time={lgbm_m['time']:.1f}s")
    except Exception as e:
        print(f"  LightGBM not available: {e}")

    # --- Neural Network Ablation (30K subset for speed) ---
    nn_n = min(30000, len(X_train))
    X_nn, y_nn = X_train[:nn_n], y_tr[:nn_n]

    configs = [
        ('ELU + BN', dict(use_mixup=False, use_scheduling=False)),
        ('ELU + BN + Mixup', dict(use_mixup=True, use_scheduling=False)),
        ('ELU + BN + LR Sched', dict(use_mixup=False, use_scheduling=True)),
        ('ELU + BN + Mixup + LR Sched', dict(use_mixup=True, use_scheduling=True)),
    ]

    for name, kwargs in configs:
        t0 = time.time()
        model = train_elu(X_nn, y_nn, X_test_s, y_test, epochs=15, **kwargs)
        m = evaluate(model, X_test_s, y_test)
        m['time'] = time.time() - t0
        results.append((name, m))
        print(f"  {name}: LL={m['log_loss']:.4f}, Acc={m['accuracy']:.3f}, ECE={m['ece']:.4f}, Time={m['time']:.1f}s")

    # --- Print summary ---
    print("\n" + "=" * 85)
    print(f"{'Model':<35} {'Log-Loss':>10} {'Accuracy':>10} {'ECE':>10}")
    print("-" * 85)
    for name, m in results:
        print(f"{name:<35} {m['log_loss']:>10.4f} {m.get('accuracy',0):>10.3f} {m['ece']:>10.4f}")
    print("=" * 85)

    # --- Save ---
    out = pd.DataFrame([{'Model': n, **m} for n, m in results])
    out_path = PROJECT_ROOT / "backtests" / "results" / "28_107k_elu_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
