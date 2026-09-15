#!/usr/bin/env python3
"""
ML QUALITY AUDIT — where does the model stand on standard metrics, and is the
agent's 0-win streak a bug or a bad model?

Checks, all strictly point-in-time (train on the N seasons before the test
season, predict the test season):

  1. Accuracy / Brier / log loss of AdaptiveMatchPredictor on the test season
     vs two baselines: always-favourite (by odds) and the market itself
     (implied probabilities from B365, margin-removed).
  2. Calibration table (predicted prob bucket -> observed frequency).
  3. The agent's bet selection: what edges it "sees", what it picks, and what
     the realised strike rate is — split by whether the pick was the market
     favourite or not.

Usage:
    python demo/ml_quality_audit.py --league I1 --train-seasons 5 --test 2526
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_sim.agent import BettingAgent, RESULT_TO_OUTCOME
from agent_sim.fetch import LEAGUES
from data.real_data import SEASON_CODES, get_season
from models.adaptive_model import AdaptiveMatchPredictor

OUTCOME_TO_RESULT = {v: k for k, v in RESULT_TO_OUTCOME.items()}


def brier_logloss(probs: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    onehot = np.eye(3)[outcomes]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    ll = -float(np.mean(np.log(np.clip(
        probs[np.arange(len(outcomes)), outcomes], 1e-12, 1.0))))
    return brier, ll


def implied_probs(o_h, o_d, o_a) -> np.ndarray:
    p = 1.0 / np.array([o_h, o_d, o_a], dtype=float)
    p = np.clip(p, 1e-4, None)
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="I1", choices=list(LEAGUES.keys()))
    ap.add_argument("--train-seasons", type=int, default=5)
    ap.add_argument("--test", default="2526", choices=SEASON_CODES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    ti = SEASON_CODES.index(args.test)
    prior = SEASON_CODES[max(0, ti - args.train_seasons):ti]
    train = pd.concat([get_season(args.league, s, args.offline)
                       for s in prior], ignore_index=True)
    train["league_code"] = args.league
    test = get_season(args.league, args.test, args.offline)
    test["league_code"] = args.league
    print(f"{LEAGUES[args.league]} {args.test}: train on {len(train)} matches "
          f"({len(prior)} seasons), predict {len(test)} matches\n")

    model = AdaptiveMatchPredictor(window=240, refit_every=60, drift_tol=0.02,
                                   min_refit=40, seed=args.seed)
    model.train(train)

    rows = []
    for _, m in test.iterrows():
        p = model.predict(m["home_team"], m["away_team"])
        probs = np.array([p["home_win"], p["draw"], p["away_win"]], dtype=float)
        market = implied_probs(m["odds_home"], m["odds_draw"], m["odds_away"])
        outcome = ["home_win", "draw", "away_win"].index(
            RESULT_TO_OUTCOME[m["result"]])
        rows.append({
            "probs": probs, "market": market, "outcome": outcome,
            "pick": ["home_win", "draw", "away_win"][int(np.argmax(probs))],
            "fav": ["home_win", "draw", "away_win"][int(np.argmin(market))],
            "pick_odds": float(m[{"home_win": "odds_home", "draw": "odds_draw",
                                  "away_win": "odds_away"}[["home_win", "draw",
                                                           "away_win"][int(np.argmax(probs))]]]),
            "max_prob": float(probs.max()),
        })

    df = pd.DataFrame(rows)
    y = df["outcome"].values
    P = np.stack(df["probs"].values)
    M = np.stack(df["market"].values)
    OUTCOMES = ["home_win", "draw", "away_win"]

    def report(name, probs):
        acc = float((probs.argmax(1) == y).mean())
        brier, ll = brier_logloss(probs, y)
        print(f"  {name:<22} acc {acc:6.1%}  Brier {brier:.4f}  log loss {ll:.4f}")

    print("\n== Test-season predictive quality ==")
    report("model", P)
    report("market (B365 implied)", M)
    fav = np.eye(3)[M.argmax(1)]
    report("always-favourite", fav)
    uniform = np.full_like(P, 1 / 3)
    report("uniform (worst case)", uniform)

    # calibration: predicted bucket mean vs realised win rate of the model pick
    print("\n== Model calibration (max-prob bucket -> realised win rate) ==")
    df["bucket"] = pd.cut(df["max_prob"], [0.33, 0.4, 0.5, 0.6, 1.0])
    hit = (P.argmax(1) == y).astype(float)
    df["hit"] = hit
    cal = df.groupby("bucket", observed=True).agg(
        n=("hit", "size"), predicted=("max_prob", "mean"),
        realised=("hit", "mean"))
    print(cal.to_string())

    # where does the model disagree with the market, and who is right?
    print("\n== Model vs market disagreements (top pick differs) ==")
    df["model_hit"] = hit
    df["fav_hit"] = (M.argmax(1) == y).astype(float)
    df["y_name"] = [OUTCOMES[i] for i in y]
    dis = df[df["pick"] != df["fav"]]
    agree = df[df["pick"] == df["fav"]]
    print(f"  agree: {len(agree)} matches, model pick right "
          f"{agree['model_hit'].mean():.1%}")
    print(f"  disagree: {len(dis)} matches, model pick right "
          f"{dis['model_hit'].mean():.1%}"
          f", market fav right {dis['fav_hit'].mean():.1%}")
    if len(dis):
        print(dis.groupby(pd.cut(dis["pick_odds"], [1.5, 2.5, 3.5, 99]),
                          observed=True)
              .agg(n=("model_hit", "size"), model_right=("model_hit", "sum"),
                   fav_right=("fav_hit", "sum"))
              .to_string())

    # now the agent-level question: what does BettingAgent actually bet?
    print("\n== Agent bet selection over the same test season ==")
    agent = BettingAgent(train, seed=args.seed)
    bets = []
    for _, m in test.iterrows():
        d = agent.decide(m, m["date"], agent.bankroll)
        agent.settle(d, m, agent.bankroll)
        if d["decision"] is not None:
            bets.append({
                "pick": d["decision"], "odds": d["odds"][d["decision"]],
                "edge": d["edge"], "threshold": d["threshold"],
                "prob": d["probs"][d["decision"]],
                "won": RESULT_TO_OUTCOME[m["result"]] == d["decision"],
                "was_fav": d["decision"] ==
                           ["home_win", "draw", "away_win"][int(np.argmin(
                               implied_probs(m["odds_home"], m["odds_draw"],
                                             m["odds_away"])))],
            })
        agent.reveal_result(m)
    b = pd.DataFrame(bets)
    if len(b):
        print(f"  bets: {len(b)}, wins {int(b.won.sum())}, "
              f"strike {b.won.mean():.1%}")
        print(f"  mean model prob on picks: {b.prob.mean():.3f}; "
              f"mean odds {b.odds.mean():.2f}; mean seen edge {b.edge.mean():+.3f} "
              f"vs threshold {b.threshold.mean():.3f}")
        print(f"  picks that were market favourite: {int(b.was_fav.sum())}/{len(b)}"
              f", strike on those {b[b.was_fav].won.mean() if b.was_fav.any() else float('nan'):.1%}")
        print(b.groupby("pick").agg(n=("won", "size"), wins=("won", "sum"),
                                    mean_odds=("odds", "mean")).to_string())
    else:
        print("  no bets placed")


if __name__ == "__main__":
    main()
