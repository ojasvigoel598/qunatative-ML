#!/usr/bin/env python3
"""
DYNAMIC THINKING LAYER — an adaptive decision engine whose MODEL is dynamic.

The layer makes every decision by combining *fresh, per-match signals* with a
base model that is ITSELF self-refitting, and it re-weights its own reasoning
from observed performance.  The code — and the model — are dynamic.

What adapts (not just the stake, the MODEL)
-------------------------------------------
1. BASE MODEL  — an `AdaptiveMatchPredictor`: PoissonElo trained on initial
   data + a Gradient-Boosting layer that REFITS on a rolling window when
   scheduled or when its rolling Brier drifts.  Elo and rolling form update
   after every revealed match.  So the probabilities themselves change as new
   information arrives (a surprise lineup / new form is baked into the next
   prediction, not left until "morning analysis").

2. MARKET SIGNALS (the "hidden signals" idea) —
   * public line (opening / soft bookmaker)  = the price you can actually get,
   * sharp line (closing / Pinnacle-style)   = a signal, not a price,
   * the public-vs-sharp SPLIT per outcome   = sharp money moving the line,
   * multi-book consensus + DISPERSION across all available bookmakers
     (high dispersion = disagreement = caution),
   * fatigue (rest days since each team last played),
   * a live-news CONDITIONS slot (injury / lineup / weather flags) — optional,
     degrades gracefully when absent (never a hard dependency).

3. ADAPTIVE REASONING — the model-vs-market blend weight is re-weighted online
   from rolling Brier: whichever source has been better calibrated lately gets
   trusted more.

4. RISK-AWARE STAKING — uncertainty (model/market disagreement + dispersion)
   and drawdown shrink the stake; a hard 2% cap; survival mode (tiny flat
   stakes, relaxed threshold) below 10% of the start.

Every decision logs a full THINKING TRACE (signals seen, weights, edges,
uncertainty, stake, why) so the simulation and videos can show *what it was
thinking*.

Usage:
    from models.dynamic_thinking import DynamicThinkingLayer
    layer = DynamicThinkingLayer(poisson, ml)          # or pass train_df
    decision = layer.think(home, away, public_odds, sharp_odds,
                           extra_books=[...], conditions={...}, current_day=d)
    layer.observe(home, away, hg, ag, result, decision, public_odds, day=d)
"""

from __future__ import annotations

import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.adaptive_model import (  # noqa: E402
    AdaptiveMatchPredictor, OnlineState, brier,
)

OUTCOMES = ("home_win", "draw", "away_win")
INDEX = {"home_win": 0, "draw": 1, "away_win": 2}
INITIAL_INVESTMENT = 1_000_000.0

EDGE_THRESHOLD = 0.03
PROB_FLOOR = 0.40
MIN_ODDS = 1.6
MIN_STAKE = 100.0
STAKE_CAP_FRAC = 0.02
BASE_STAKE = 10_000.0
SURVIVAL_FLOOR = 0.10
SURVIVAL_STAKE_FRAC = 0.005
SURVIVAL_EDGE = 0.01

WEIGHT_WINDOW = 80
MAX_DISAGREEMENT = 0.25
CONF_DRIFT_TOL = 0.08     # refit the base when rolling confidence drops this far
CONF_REFIT_GAP = 50       # min matches between confidence-gated refits


def implied_probs(odds: dict) -> np.ndarray:
    """Normalised inverse odds -> implied probabilities vector (H, D, A)."""
    p = np.array([1.0 / odds[o] for o in OUTCOMES], dtype=float)
    p = np.clip(p, 1e-4, None)
    return p / p.sum()


class DynamicThinkingLayer:
    """Self-refitting base model + multi-signal fusion + adaptive staking."""

    def __init__(self, poisson=None, ml=None, train_df: pd.DataFrame = None,
                 bankroll: float = INITIAL_INVESTMENT, seed: int = 42,
                 base_kwargs: dict = None, confidence_aware: bool = True,
                 simple: bool = False):
        """
        Args:
            poisson/ml:     pre-trained base ensemble (used when train_df is None
                            OR when simple=True).
            train_df:       if given, the layer owns a self-refitting
                            AdaptiveMatchPredictor (the FULLY dynamic base).
            confidence_aware: use confidence to modulate the calibration
                            blend, gate base refits, and scale stakes (the
                            confidence-aware adaptation upgrade).
            simple:         reproduce the ORIGINAL (v1) layer semantics: fixed
                            poisson+ml base, NO self-refit, NO multi-book
                            consensus/dispersion, NO conditions, plain Brier
                            blend, no confidence scaling.  Used as the honest
                            baseline when measuring what each upgrade buys.
        """
        self.confidence_aware = confidence_aware
        self.simple = simple

        # The base model is dynamic: if we're given training data we build an
        # AdaptiveMatchPredictor that refits itself online.  Otherwise fall
        # back to a PoissonElo(+GB) base (Elo/form still update online).
        if train_df is not None and not simple:
            kwargs = {"window": 200, "refit_every": 100,
                      "drift_tol": 0.03, "min_refit": 100}
            if base_kwargs:
                kwargs.update(base_kwargs)
            self.base = AdaptiveMatchPredictor(static=False, **kwargs)
            self.base.train(train_df)
        else:
            from models.poisson_elo_model import PoissonEloModel
            self.poisson = poisson if poisson is not None else PoissonEloModel()
            self.ml = ml
            self.base = None

        self.state = OnlineState()
        self.last_played = {}          # team -> last match day (fatigue signal)
        self.start = float(bankroll)
        self.bankroll = self.start
        self.peak = self.start
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._model_brier = deque(maxlen=WEIGHT_WINDOW)
        self._market_brier = deque(maxlen=WEIGHT_WINDOW)
        self._conf_hist = deque(maxlen=WEIGHT_WINDOW)  # per-match confidence
        self._best_conf = 0.0
        self._last_conf_refit = 0
        self.conf_refits = 0
        self.n_obs = 0
        self.market_weight = 0.5
        self.n_bets = 0
        self.n_wins = 0
        self.trace = []
        self.survival_active = False
        self.survival_triggered_at = None

    # ------------------------------------------------------------ confidence
    @staticmethod
    def _confidence(probs: np.ndarray) -> float:
        """Margin-based confidence in [0, 1].

        How far the top outcome's probability sits above the uniform 1/3:
        conf = (p_max - 1/3) / (2/3).  1.0 = certain, 0.0 = coin-flip.
        (Entropy is NOT used: for football-sized probabilities it saturates
        near zero and would make the layer's confidence almost constant,
        which defeats confidence-aware adaptation.)
        """
        p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
        p = p / p.sum()
        return float(np.clip((p.max() - 1 / 3) / (2 / 3), 0.0, 1.0))

    # ------------------------------------------------------------ base probs
    def _base_probs(self, home: str, away: str) -> np.ndarray:
        if self.base is not None:
            p = self.base.predict(home, away)
        else:
            p = self.poisson.predict(home, away)
            if self.ml is not None:
                m = self.ml.predict_proba(
                    home, away,
                    home_elo=self.poisson.get_team_elo(home),
                    away_elo=self.poisson.get_team_elo(away))
                p = {k: 0.6 * p[k] + 0.4 * m[k] for k in OUTCOMES}
        return np.array([p["home_win"], p["draw"], p["away_win"]], dtype=float)

    def _rest_days(self, home: str, away: str, current_day: int) -> tuple:
        rh = current_day - self.last_played.get(home, current_day - 6)
        ra = current_day - self.last_played.get(away, current_day - 6)
        return max(rh, 0), max(ra, 0)

    # ------------------------------------------------------------ the decision
    def think(self, home: str, away: str, public_odds: dict, sharp_odds: dict,
              extra_books: list = None, conditions: dict = None,
              current_day: int = 0) -> dict:
        """Fuse ALL available signals and return a decision + thinking trace."""
        model_p = self._base_probs(home, away)
        sharp_p = implied_probs(sharp_odds)
        public_p = implied_probs(public_odds)
        market_split = sharp_p - public_p            # hidden signal per outcome
        rest_h, rest_a = self._rest_days(home, away, current_day)

        if self.simple:
            # v1 semantics: no multi-book fusion, no dispersion, no conditions
            consensus = sharp_p
            dispersion = 0.0
            cond_note = ""
        else:
            # --- multi-book consensus + dispersion (hidden signal #2)
            books = [public_p, sharp_p]
            if extra_books:
                for b in extra_books:
                    try:
                        books.append(implied_probs(b))
                    except Exception:
                        pass
            book_mat = np.array(books)
            consensus = book_mat.mean(axis=0)
            dispersion = float(book_mat.std(axis=0).mean())  # disagreement across books
            cond_note = ""

        # --- adaptive blend: self-refitting model vs sharp market
        w = float(np.clip(self.market_weight, 0.0, 0.8))
        fused = (1 - w) * model_p + w * consensus
        fused = np.clip(fused / fused.sum(), 1e-6, 1.0)
        fused = fused / fused.sum()

        if not self.simple and conditions:
            for outcome, delta in conditions.items():
                if outcome in INDEX:
                    fused[INDEX[outcome]] = max(1e-4, fused[INDEX[outcome]] + delta)
            fused = fused / fused.sum()
            cond_note = f" · cond:{','.join(f'{k}{v:+.0%}' for k, v in conditions.items())}"

        # --- value vs the PUBLIC price (the price you can actually get)
        edges = {o: float(fused[INDEX[o]] * public_odds[o] - 1.0) for o in OUTCOMES}
        best = max(edges, key=edges.get)
        edge = edges[best]

        # --- CONFIDENCE of the fused distribution (the adaptation signal)
        conf = self._confidence(fused)
        # stake scale: how far the CHOSEN outcome's probability sits above the
        # 0.40 minimum-pass floor.  At the floor -> 1.0 (identical to the
        # previous layer); more sure -> scaled up to 1.4; never below 0.75.
        if self.confidence_aware:
            conf_factor = float(np.clip(fused[INDEX[best]] / PROB_FLOOR, 0.75, 1.4))
        else:
            conf_factor = 1.0

        # --- uncertainty: model-vs-market disagreement + book dispersion
        disagreement = float(np.abs(model_p - sharp_p).mean())
        uncertainty = float(1.0 - fused.max())
        shrink = float(np.clip(
            1.0 - disagreement / MAX_DISAGREEMENT - dispersion * 0.4, 0.15, 1.0))

        # --- dynamic risk factor from drawdown
        dd = (self.peak - self.bankroll) / self.peak
        risk_factor = float(np.clip(1.0 - 2.0 * dd, 0.25, 1.0))

        # --- stake (confidence-aware: commit more when the layer is sure)
        stake = 0.0
        decision = None
        if best and edge > (SURVIVAL_EDGE if self.survival_active else EDGE_THRESHOLD):
            prob_ok = self.survival_active or fused[INDEX[best]] >= PROB_FLOOR
            odds = public_odds[best]
            if prob_ok and odds >= MIN_ODDS:
                if self.survival_active:
                    stake = SURVIVAL_STAKE_FRAC * self.bankroll
                else:
                    stake = min(BASE_STAKE * shrink * risk_factor * conf_factor,
                                STAKE_CAP_FRAC * self.bankroll)
                if stake >= MIN_STAKE:
                    decision = best

        trace = {
            "match": f"{home} vs {away}",
            "model_p": model_p.round(3),
            "sharp_p": sharp_p.round(3),
            "public_p": public_p.round(3),
            "consensus": consensus.round(3),
            "dispersion": round(dispersion, 3),
            "market_split": market_split.round(3),
            "market_weight": round(w, 3),
            "rest_days": (rest_h, rest_a),
            "fused": fused.round(3),
            "disagreement": round(disagreement, 3),
            "uncertainty": round(uncertainty, 3),
            "confidence": round(conf, 3),
            "conf_factor": round(conf_factor, 3),
            "risk_factor": round(risk_factor, 3),
            "edges": {k: round(v, 3) for k, v in edges.items()},
            "decision": decision,
            "edge": round(edge, 3),
            "stake": round(stake, 2),
            "survival": self.survival_active,
            "conditions": cond_note.strip(" ·"),
        }
        self.trace.append(trace)
        return trace

    # ------------------------------------------------------------ reveal result
    def observe(self, home: str, away: str, hg: float, ag: float,
                result: str, decision: dict, public_odds: dict,
                current_day: int = 0):
        """Reveal the result: bankroll, online state, calibration, survival."""
        # 1. resolve the bet
        outcome = decision["decision"]
        if outcome is not None:
            odds = public_odds[outcome]
            won = (result == "H" and outcome == "home_win") or \
                  (result == "D" and outcome == "draw") or \
                  (result == "A" and outcome == "away_win")
            stake = decision["stake"]
            profit = stake * (odds - 1.0) if won else -stake
            self.bankroll += profit
            self.peak = max(self.peak, self.bankroll)
            self.n_bets += 1
            self.n_wins += int(won)
            decision["win"] = bool(won)
            decision["profit"] = round(profit, 2)

        # 2. online state + self-refitting base model
        if self.base is not None:
            # reveal to the AdaptiveMatchPredictor (Elo/form + drift refit)
            self.base.observe(home, away, float(hg), float(ag), result)
        else:
            self.state.update(home, away, float(hg), float(ag), result)
        self.last_played[home] = current_day
        self.last_played[away] = current_day

        # 3. calibration vs sharp market on THIS match
        y_true = 0 if result == "H" else (1 if result == "D" else 2)
        onehot = np.zeros(3); onehot[y_true] = 1.0
        conf = float(decision.get("confidence", 0.5))
        self.n_obs += 1
        self._model_brier.append(float(np.mean((decision["model_p"] - onehot) ** 2)))
        self._market_brier.append(float(np.mean((decision["sharp_p"] - onehot) ** 2)))
        self._conf_hist.append(conf)

        # 4. ADAPT the blend weight — CONFIDENCE-AWARE:
        #    * the calibration comparison is weighted by confidence, so the
        #      blend is judged mostly on the decisions the layer was SURE
        #      about (where it actually commits money);
        #    * the weight moves toward the target with a step that grows
        #      with confidence, i.e. it adapts faster when it has a clear
        #      signal, more cautiously when it is guessing.
        if len(self._model_brier) >= 30:
            cs = np.clip(np.array(self._conf_hist), 0.05, None)
            if cs.sum() > 0:
                mb = float(np.average(np.array(self._model_brier), weights=cs))
                kb = float(np.average(np.array(self._market_brier), weights=cs))
            else:
                mb = float(np.mean(self._model_brier))
                kb = float(np.mean(self._market_brier))
            if mb + kb > 1e-9:
                target = float(np.clip(kb / (mb + kb), 0.0, 0.8))
                if self.confidence_aware:
                    step = float(np.clip(0.15 + 0.45 * conf, 0.15, 0.6))
                    self.market_weight = float((1 - step) * self.market_weight
                                               + step * target)
                else:
                    self.market_weight = target

        # 4b. CONFIDENCE-GATED base refit: if the base model's rolling
        #     confidence has decayed vs its best, it is losing its grip on
        #     the world — re-learn from the recent window immediately
        #     (in addition to the scheduled/drift refits inside the base).
        if (self.base is not None and self.confidence_aware
                and len(self._conf_hist) >= 30):
            rc = float(np.mean(self._conf_hist))
            self._best_conf = max(self._best_conf, rc)
            if (rc < self._best_conf - CONF_DRIFT_TOL
                    and self.n_obs - self._last_conf_refit >= CONF_REFIT_GAP):
                self.base.refit_now()
                self._last_conf_refit = self.n_obs
                self.conf_refits += 1

        # 5. survival mode
        if self.bankroll < SURVIVAL_FLOOR * self.start:
            if not self.survival_active:
                self.survival_active = True
                self.survival_triggered_at = current_day

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict:
        return {
            "final_bankroll": round(self.bankroll, 2),
            "roi_pct": round((self.bankroll / self.start - 1) * 100, 2),
            "n_bets": self.n_bets,
            "n_wins": self.n_wins,
            "strike_rate": round(self.n_wins / self.n_bets * 100, 2) if self.n_bets else 0.0,
            "final_market_weight": round(self.market_weight, 3),
            "base_refits": getattr(self.base, "refits", 0),
            "conf_refits": self.conf_refits,
            "final_confidence": round(float(np.mean(self._conf_hist)), 3)
            if self._conf_hist else 0.0,
            "confidence_aware": self.confidence_aware,
            "survival": self.survival_active,
            "survival_day": self.survival_triggered_at,
        }


if __name__ == "__main__":
    # smoke test with a self-refitting base (the truly dynamic path)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "home_team": rng.choice(["A", "B", "C", "D"], 400),
        "away_team": rng.choice(["A", "B", "C", "D"], 400),
        "home_goals": rng.poisson(1.6, 400),
        "away_goals": rng.poisson(1.2, 400),
    })
    df = df[df["home_team"] != df["away_team"]].reset_index(drop=True)
    df["result"] = np.where(df["home_goals"] > df["away_goals"], "H",
                            np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    layer = DynamicThinkingLayer(train_df=df.iloc[:300], bankroll=10_000.0)
    dec = layer.think("A", "B", {"home_win": 2.0, "draw": 3.4, "away_win": 4.0},
                      {"home_win": 1.9, "draw": 3.5, "away_win": 4.4},
                      extra_books=[{"home_win": 2.05, "draw": 3.3, "away_win": 4.1}],
                      conditions={"home_win": -0.02}, current_day=1)
    print("decision:", {k: dec[k] for k in
                        ("decision", "edge", "stake", "market_weight",
                         "dispersion", "rest_days", "confidence", "conf_factor")})
    layer.observe("A", "B", 2, 1, "H", dec,
                  {"home_win": 2.0, "draw": 3.4, "away_win": 4.0}, current_day=1)
    print("summary:", layer.summary())

    # v1 (simple) mode: fixed poisson+ml base, no confidence, no fusion
    from models.poisson_elo_model import PoissonEloModel
    poisson = PoissonEloModel()
    poisson.train(df.iloc[:300])
    v1 = DynamicThinkingLayer(poisson=poisson, ml=None, bankroll=10_000.0,
                              simple=True, confidence_aware=False)
    d1 = v1.think("A", "B", {"home_win": 2.0, "draw": 3.4, "away_win": 4.0},
                  {"home_win": 1.9, "draw": 3.5, "away_win": 4.4}, current_day=1)
    assert d1["dispersion"] == 0.0 and d1["conditions"] == ""
    assert v1.base is None
    print("v1 decision:", {k: d1[k] for k in ("decision", "edge", "stake",
                                               "confidence", "dispersion")})
    print("[OK] DynamicThinkingLayer (confidence-aware + v1) smoke test passed.")
