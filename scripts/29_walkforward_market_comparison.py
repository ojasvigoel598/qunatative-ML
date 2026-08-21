#!/usr/bin/env python3
"""
29 — Walk-Forward Market Comparison with Rolling Team Form

Tests whether adding pre-match rolling features gives edge over bookmaker odds.
Walk-forward validation ensures no leakage. Tests each league separately.

Features (ALL pre-match, shift(1) ensures no leakage):
  - Bookmaker odds, implied probabilities, overround
  - Home/Away team last-5 win rate
  - Home/Away team last-10 goal average
  - ELO ratings
  - H2H record

Usage:
    python scripts/29_walkforward_market_comparison.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_rolling_features_vectorized(df, window=5):
    """Rolling team form via pandas groupby. shift(1) prevents leakage."""
    records = []
    for i, row in df.iterrows():
        records.append({'idx': i, 'team': row['home_team'], 'is_home': 1,
                        'gf': row['home_goals'], 'ga': row['away_goals'], 'result': row['result']})
        records.append({'idx': i, 'team': row['away_team'], 'is_home': 0,
                        'gf': row['away_goals'], 'ga': row['home_goals'], 'result': row['result']})

    expanded = pd.DataFrame(records).sort_values(['team', 'idx'])
    expanded['win'] = expanded.apply(
        lambda r: 1.0 if (r['is_home'] and r['result'] == 'H') or
                          (not r['is_home'] and r['result'] == 'A') else
                  0.5 if r['result'] == 'D' else 0.0, axis=1)

    expanded['wr'] = expanded.groupby('team')['win'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    expanded['gf_avg'] = expanded.groupby('team')['gf'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    expanded['ga_avg'] = expanded.groupby('team')['ga'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())

    home_exp = expanded[expanded['is_home'] == 1].set_index('idx')
    away_exp = expanded[expanded['is_home'] == 0].set_index('idx')

    return (home_exp['wr'].reindex(df.index).fillna(0.5).values,
            home_exp['gf_avg'].reindex(df.index).fillna(1.3).values,
            home_exp['ga_avg'].reindex(df.index).fillna(1.3).values,
            away_exp['wr'].reindex(df.index).fillna(0.5).values,
            away_exp['gf_avg'].reindex(df.index).fillna(1.3).values,
            away_exp['ga_avg'].reindex(df.index).fillna(1.3).values)


def compute_elo_ratings(df, k=20, home_adv=50):
    elos = {}
    h_elo_l, a_elo_l = [], []
    for _, row in df.iterrows():
        h = row['home_team']; a = row['away_team']
        h_e = elos.get(h, 1500); a_e = elos.get(a, 1500)
        h_elo_l.append(h_e); a_elo_l.append(a_e)
        h_exp = 1 / (1 + 10 ** ((a_e - h_e - home_adv) / 400))
        s = {'H': (1.0, 0.0), 'A': (0.0, 1.0)}.get(row['result'], (0.5, 0.5))
        elos[h] = h_e + k * (s[0] - h_exp)
        elos[a] = a_e + k * (s[1] - (1 - h_exp))
    return np.array(h_elo_l), np.array(a_elo_l)


def compute_h2h(df, window=10):
    rates = np.full(len(df), 0.33)
    history = {}
    for i, row in df.iterrows():
        pair = tuple(sorted([row['home_team'], row['away_team']]))
        hist = history.get(pair, [])[-window:]
        if hist:
            hw = sum(1 for t, r in hist if t == row['home_team'] and r == 'H')
            hw += sum(1 for t, r in hist if t == row['away_team'] and r == 'A')
            hw += 0.5 * sum(1 for _, r in hist if r == 'D')
            rates[i] = hw / len(hist)
        history.setdefault(pair, []).append((row['home_team'], row['result']))
    return rates


def build_features(df):
    print("  Computing rolling form (last 5)...")
    h_wr5, h_gf5, h_ga5, a_wr5, a_gf5, a_ga5 = compute_rolling_features_vectorized(df, 5)
    print("  Computing rolling form (last 10)...")
    h_wr10, h_gf10, h_ga10, a_wr10, a_gf10, a_ga10 = compute_rolling_features_vectorized(df, 10)
    print("  Computing ELO...")
    h_elo, a_elo = compute_elo_ratings(df)
    print("  Computing H2H...")
    h2h = compute_h2h(df)

    odds_h = df['odds_home'].clip(lower=1.01).values
    odds_d = df['odds_draw'].clip(lower=1.01).values
    odds_a = df['odds_away'].clip(lower=1.01).values
    inv_h, inv_d, inv_a = 1 / odds_h, 1 / odds_d, 1 / odds_a
    total = inv_h + inv_d + inv_a

    X = np.column_stack([
        odds_a, odds_d, odds_h,
        inv_a / total, inv_d / total, inv_h / total, total - 1,
        h_wr5, h_gf5, h_ga5, h_wr10, h_gf10, h_ga10,
        a_wr5, a_gf5, a_ga5, a_wr10, a_gf10, a_ga10,
        h_elo / 2000, a_elo / 2000, h2h,
    ])
    names = ['odds_a', 'odds_d', 'odds_h', 'impl_a', 'impl_d', 'impl_h', 'overround',
             'h_wr5', 'h_gf5', 'h_ga5', 'h_wr10', 'h_gf10', 'h_ga10',
             'a_wr5', 'a_gf5', 'a_ga5', 'a_wr10', 'a_gf10', 'a_ga10',
             'h_elo', 'a_elo', 'h2h']
    return X, names


def main():
    print("=" * 80)
    print("WALK-FORWARD MARKET COMPARISON — 14 LEAGUES, 107K MATCHES")
    print("=" * 80)

    # Load
    t0 = time.time()
    df = pd.read_csv(PROJECT_ROOT / "data" / "real" / "all_leagues_combined.csv", low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['home_goals', 'away_goals', 'result', 'odds_home', 'odds_draw', 'odds_away'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  {len(df)} matches, {df['league'].nunique()} leagues ({time.time()-t0:.1f}s)")

    y_all = df['result'].map({'H': 2, 'A': 0, 'D': 1}).values

    print("\nBuilding pre-match features...")
    t0 = time.time()
    X_all, feature_names = build_features(df)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=2.0, neginf=0.0)
    print(f"  {len(feature_names)} features ({time.time()-t0:.1f}s)")

    # Per-league walk-forward
    print("\nWalk-forward per league:")
    print("-" * 105)
    print(f"{'League':<25} {'ModelLL':>8} {'MktLL':>8} {'Edge':>7} {'Bets':>5} {'Win%':>5} {'ROI%':>7} {'P/L':>10} {'DD%':>5}")
    print("-" * 105)

    all_results = []
    for league in sorted(df['league'].unique()):
        mask = df['league'] == league
        idx = np.where(mask)[0]
        if len(idx) < 500:
            continue

        df_l = df.iloc[idx].reset_index(drop=True)
        X_l = X_all[idx]
        y_l = y_all[idx]
        n = len(df_l)

        # Walk-forward: 50% train, then 10% steps
        train_end = n // 2
        step = n // 10

        all_probs, all_true, all_test_idx = [], [], []
        i = train_end
        while i + step <= n:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_l[:i])
            X_te = scaler.transform(X_l[i:i + step])

            model = LogisticRegression(max_iter=1000)
            model.fit(X_tr, y_l[:i])
            probs = model.predict_proba(X_te)

            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            probs = probs / probs.sum(axis=1, keepdims=True)

            all_probs.append(probs)
            all_true.append(y_l[i:i + step])
            all_test_idx.extend(range(i, i + step))
            i += step

        probs_arr = np.vstack(all_probs)
        y_test = np.concatenate(all_true)
        test_idx = np.array(all_test_idx)

        # Model log-loss
        model_ll = log_loss(y_test, probs_arr)

        # Market log-loss (matching the SAME test matches)
        odds_h = df_l['odds_home'].iloc[test_idx].clip(lower=1.01).values
        odds_d = df_l['odds_draw'].iloc[test_idx].clip(lower=1.01).values
        odds_a = df_l['odds_away'].iloc[test_idx].clip(lower=1.01).values
        inv_h, inv_d, inv_a = 1/odds_h, 1/odds_d, 1/odds_a
        total_imp = inv_h + inv_d + inv_a
        mkt = np.column_stack([inv_a / total_imp, inv_d / total_imp, inv_h / total_imp])
        market_ll = log_loss(y_test, mkt)

        edge_ll = market_ll - model_ll

        # ROI: bet when model has edge over market
        bankroll = 10000.0
        initial = bankroll
        bets = wins = 0
        total_staked = profit = 0.0
        peak = bankroll
        max_dd = 0.0

        for j in range(len(y_test)):
            pred = np.argmax(probs_arr[j])
            model_p = probs_arr[j][pred]
            market_p = mkt[j][pred]
            edge = model_p - market_p

            if edge > 0.02 and model_p > 0.35:
                odds_key = {0: 'away', 1: 'draw', 2: 'home'}[pred]
                odds_val = df_l['odds_' + odds_key].iloc[test_idx[j]]
                if pd.isna(odds_val) or odds_val < 1.01:
                    continue
                odds_val = max(odds_val, 1.01)

                kelly = (model_p * (odds_val - 1) - (1 - model_p)) / (odds_val - 1)
                stake = bankroll * 0.25 * max(0, kelly)
                stake = min(stake, bankroll * 0.02)  # max 2% per bet
                stake = min(stake, 500)

                if stake > 1:
                    total_staked += stake
                    bets += 1
                    if y_test[j] == pred:
                        bankroll += stake * (odds_val - 1)
                        wins += 1
                        profit += stake * (odds_val - 1)
                    else:
                        bankroll -= stake
                        profit -= stake
                    peak = max(peak, bankroll)
                    max_dd = max(max_dd, (peak - bankroll) / peak)

        wr = wins / bets * 100 if bets > 0 else 0
        roi = profit / total_staked * 100 if total_staked > 0 else 0

        result = {
            'league': league, 'matches': n, 'test': len(y_test),
            'model_ll': model_ll, 'market_ll': market_ll, 'edge': edge_ll,
            'bets': bets, 'win_rate': wr, 'roi': roi, 'profit': profit,
            'max_dd': max_dd * 100,
        }
        all_results.append(result)

        print(f"{league:<25} {model_ll:>8.4f} {market_ll:>8.4f} {edge_ll:>+7.4f} "
              f"{bets:>5} {wr:>4.1f}% {roi:>+6.1f}% {profit:>+9.0f} {max_dd*100:>4.1f}%")

    print("-" * 105)

    # Aggregate
    tb = sum(r['bets'] for r in all_results)
    tp = sum(r['profit'] for r in all_results)
    ml = np.mean([r['model_ll'] for r in all_results])
    mk = np.mean([r['market_ll'] for r in all_results])
    me = np.mean([r['edge'] for r in all_results])

    print(f"{'AVERAGE':<25} {ml:>8.4f} {mk:>8.4f} {me:>+7.4f} {tb:>5}")
    print(f"\n  Total bets: {tb}")
    print(f"  Total profit: ${tp:+,.0f}")
    print(f"  Avg LL Edge: {me:+.4f} (positive = model better)")

    # Only bet on markets where model beats market
    beat = [r for r in all_results if r['edge'] > 0 and r['roi'] > 0]
    if beat:
        print(f"\n  Leagues where model beats market: {len(beat)}/{len(all_results)}")
        for r in beat:
            print(f"    {r['league']}: edge={r['edge']:+.4f}, ROI={r['roi']:+.1f}%")

    # Save
    out = pd.DataFrame(all_results)
    out_path = PROJECT_ROOT / "backtests" / "results" / "29_walkforward_market_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
