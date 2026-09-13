#!/usr/bin/env python3
"""
AI + ML AGENT — an AI reasoning layer wrapped around the ML betting agent.

Why an AI layer on top of the ML model?
---------------------------------------
The project's own results (backtests/results/) show exactly where the pure-ML
agent loses and where the ONLY positive-ROI signals live:

  * why_model_losing.txt        — the winner's curse (advertised edge +14% vs
                                  realised +3%) is the DOMINANT loss mechanism.
  * hidden_signals_results.csv  — the sharp-vs-public split is the money
                                  finding: +1.16% CLV per bet (t=4.72, p<0.001,
                                  94/200 positive) on real matches.
  * clv_focused_results.csv     — CLV-gated betting on real 25/26 seasons:
                                  +53.2% (La Liga), +14.8% (Serie A) — the
                                  only large positive-ROI configs in the repo,
                                  achieved on ONE season of data (small scale).
  * multi_run_aggregate.csv     — the ML-only agent: mean ROI -16.9%, 0/3 runs
                                  profitable.

So the AI layer does NOT try to out-predict the market.  It manages the ML
model's bets the way a professional risk desk would, using the three signals
the project has already proven carry real information:

  1. SHARP-VS-PUBLIC SPLIT  — only let the ML bet through when the sharp line
     (Pinnacle) agrees with the direction the ML is backing at the public
     price.  This is the signal with the strongest measured CLV.
  2. CLV GATE               — require the bet's expected CLV vs the sharp
     closing line to be positive (the standard test of real information).
  3. DISPERSION CAUTION     — shrink stakes when books disagree (and never
     size up when the ML's disagreement with the market is largest — that is
     exactly where the winner's curse bites).

The AI never sees the future: every signal it uses is computed from odds
available at prediction time.  It can VETO a bet, DOWNSIZE it, or (when the
signals are strongly aligned) UPSCALE it within a hard cap.

Usage:
    from agent_sim.ai_agent import AIMLBettingAgent
    agent = AIMLBettingAgent(train_df, seed=42)
    trace = agent.decide(match, sim_day, bankroll)   # same interface as BettingAgent
    agent.reveal_result(match); agent.settle(trace, match, bankroll)
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_sim.agent import BettingAgent, RESULT_TO_OUTCOME

# ---- AI policy constants (all learned/justified from the repo's own results)
#
# Evidence base (backtests/results/):
#  * clv_threshold_results.csv   — raising the ML EDGE threshold makes ROI
#    WORSE (-42% @0% -> -45% @4%): model-edge selection IS the winner's curse.
#  * clv_focused_results.csv     — the only big positive-ROI configs (+53.2%
#    La Liga, +14.8% Serie A) bet only where OUR price beats the SHARP price
#    (expected CLV > 0) at avg odds ~2.1 — never on longshots.
#  * data_size_sweep_report.md   — on small training data the ML's failures
#    concentrate in longshots (odds >= 3.5 lose ~100% in the walk-forward).
SPLIT_MIN_EDGE = 0.03      # veto if the sharp line moves AGAINST the pick
CLV_GATE = 0.00            # our price must beat the sharp line (expected CLV >= 0)
DISPERSION_CAP = 0.12      # above this book disagreement -> veto
MAX_ODDS_GUARD = 3.5       # AI longshot guard: no bets at or above these odds
AGREE_UPSCALE = 1.25       # strong agreement -> 25% bigger stake
DOWNSIZE_FLOOR = 0.25      # never stake less than 25% of the ML stake
MAX_STAKE_CAP = 10_000.0   # hard AI cap (same as the ML flat default)
SHRINK_WITH_DISAGREEMENT = 4.0  # how fast stakes shrink as ML-vs-market gap grows


class AIMLBettingAgent(BettingAgent):
    """The ML agent + an AI veto/size layer using proven market signals.

    The ML base (AdaptiveMatchPredictor) is inherited unchanged — the AI layer
    only manages WHICH ML bets pass and HOW BIG they are.
    """

    name = "AI+ML"

    def __init__(self, *args, ai_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        # per-instance overrides let research sweeps test gate variants without
        # touching the calibrated defaults
        cfg = {"SPLIT_MIN_EDGE": SPLIT_MIN_EDGE, "CLV_GATE": CLV_GATE,
               "DISPERSION_CAP": DISPERSION_CAP,
               "MAX_ODDS_GUARD": MAX_ODDS_GUARD,
               "AGREE_UPSCALE": AGREE_UPSCALE}
        if ai_config:
            cfg.update(ai_config)
        self.cfg = cfg
        self.ai_vetoes = 0
        self.ai_passes = 0
        self.ai_upscales = 0
        self._ai_log = deque(maxlen=500)

    # ------------------------------------------------------------- signals
    @staticmethod
    def _implied(odds: dict) -> np.ndarray:
        p = np.array([odds["home_win"], odds["draw"], odds["away_win"]],
                     dtype=float)
        p = np.clip(p, 1e-4, None)
        return p / p.sum()

    def _ai_review(self, probs: dict, odds: dict, sharp: dict, best: str,
                   edge: float, stake: float) -> tuple[float, str, dict]:
        """AI verdict on an ML bet: (final_stake, reason, ai_trace)."""
        outcome_to_idx = {"home_win": 0, "draw": 1, "away_win": 2}
        i = outcome_to_idx[best]

        market_p = self._implied(odds)
        sharp_vals = [sharp[k] for k in sharp]
        sharp_ok = all(v is not None and np.isfinite(v) and v > 1.0
                       for v in sharp_vals)
        sharp_p = self._implied(sharp) if sharp_ok else market_p

        # signal 1 — sharp-vs-public split: does the sharp line agree with the
        # direction the ML is backing at the public price?
        split = float(sharp_p[i] - market_p[i])

        # signal 2 — expected CLV vs the sharp line (the honest information test).
        # Standard CLV: (my price / sharp price) - 1.  Positive = we are getting
        # a better price than the sharp close, i.e. real information.
        so = sharp.get(best)
        exp_clv = float(odds[best] / so - 1.0) \
            if (so is not None and np.isfinite(so) and so > 1.0) else -1.0

        # signal 3 — dispersion across the two price sources
        dispersion = float(np.abs(market_p - sharp_p).mean())

        # signal 4 — winner's-curse guard: how far the ML is from the market
        disagreement = float(np.abs(probs[best] - market_p[i]))

        reasons = []
        cfg = self.cfg
        if odds[best] >= cfg["MAX_ODDS_GUARD"]:
            return 0.0, (f"AI VETO: longshot guard (odds {odds[best]:.2f} "
                         f">= {cfg['MAX_ODDS_GUARD']:.1f})"), {}
        if split < -cfg["SPLIT_MIN_EDGE"]:
            return 0.0, f"AI VETO: sharp line against pick (split {split:+.1%})", {}
        if dispersion > cfg["DISPERSION_CAP"]:
            return 0.0, f"AI VETO: book dispersion {dispersion:.1%} too high", {}
        if exp_clv < cfg["CLV_GATE"]:
            return 0.0, (f"AI VETO: expected CLV {exp_clv:+.1%} < "
                         f"{cfg['CLV_GATE']:+.1%}"), {}

        # size: shrink with ML-vs-market disagreement (winner's curse guard),
        # grow modestly when sharp money agrees with the ML pick.
        shrink = float(np.clip(1.0 - disagreement * SHRINK_WITH_DISAGREEMENT,
                               DOWNSIZE_FLOOR, 1.0))
        final = stake * shrink
        if split > cfg["SPLIT_MIN_EDGE"] and exp_clv > 0.01:
            final = min(final * cfg["AGREE_UPSCALE"], MAX_STAKE_CAP,
                        0.02 * self.bankroll)
            reasons.append("sharp agrees -> upscaled")
            self.ai_upscales += 1
        else:
            reasons.append(f"shrunk x{shrink:.2f} (ML-market gap {disagreement:.0%})")
        final = float(np.clip(final, 0.0, min(MAX_STAKE_CAP, 0.02 * self.bankroll)))
        if final < 1.0:
            return 0.0, "AI VETO: stake below minimum after sizing", {}

        self.ai_passes += 1
        reasons.insert(0, f"AI PASS (split {split:+.1%}, clv {exp_clv:+.1%}, "
                           f"odds {odds[best]:.2f})")
        trace = {"split": round(split, 4), "exp_clv": round(exp_clv, 4),
                 "dispersion": round(dispersion, 4),
                 "disagreement": round(disagreement, 4)}
        return final, "; ".join(reasons), trace

    # ------------------------------------------------------------- decision
    def decide(self, match: pd.Series, sim_day, bankroll_before: float) -> dict:
        trace = super().decide(match, sim_day, bankroll_before)
        if trace["decision"] is None:
            return trace                       # ML said no-bet: AI adds nothing

        best = trace["decision"]
        odds = {k: float(v) for k, v in trace["odds"].items()}
        sharp = {k: float(v) if v is not None else np.nan
                 for k, v in trace["sharp"].items()}
        probs = {k: float(v) for k, v in trace["probs"].items()}
        ml_stake = trace["stake"]

        final_stake, ai_reason, ai_trace = self._ai_review(
            probs, odds, sharp, best, trace["edge"], ml_stake)

        if final_stake <= 0.0:
            self.ai_vetoes += 1
            trace["stake"] = 0.0
            trace["decision"] = None
            trace["reason"] = ai_reason
            trace["ai_layer"] = "VETO"
        else:
            trace["stake"] = round(final_stake, 2)
            trace["reason"] = f"{trace['reason']} | {ai_reason}"
            trace["ai_layer"] = "PASS"
        trace["ai"] = ai_trace
        self._ai_log.append({k: trace.get(k) for k in
                             ("match", "decision", "stake", "ai_layer", "reason")})
        return trace

    # -------------------------------------------------------------- summary
    def summary(self) -> dict:
        s = super().summary()
        s.update({
            "agent_type": self.name,
            "ai_vetoes": self.ai_vetoes,
            "ai_passes": self.ai_passes,
            "ai_upscales": self.ai_upscales,
        })
        return s


if __name__ == "__main__":
    # smoke test on synthetic data (same pattern as models/adaptive_model.py)
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(12)]
    n = 400
    df = pd.DataFrame({
        "home_team": rng.choice(teams, n), "away_team": rng.choice(teams, n),
        "home_goals": rng.poisson(1.6, n), "away_goals": rng.poisson(1.2, n),
    })
    df = df[df["home_team"] != df["away_team"]].reset_index(drop=True)
    df["result"] = np.where(df["home_goals"] > df["away_goals"], "H",
                            np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    match = pd.Series({
        "home_team": "T0", "away_team": "T1", "result": "H",
        "league_code": "I1", "league": "Serie A", "date": pd.Timestamp("2025-01-01"),
        "odds_home": 2.0, "odds_draw": 3.4, "odds_away": 4.0,
        "pin_home": 1.9, "pin_draw": 3.5, "pin_away": 4.4,
    })
    agent = AIMLBettingAgent(df, bankroll=100_000.0, seed=0)
    d = agent.decide(match, pd.Timestamp("2025-01-01"), agent.bankroll)
    print("decision:", d["decision"], "| stake:", d["stake"])
    print("reason:", d["reason"])
    agent.settle(d, match, agent.bankroll)
    print("summary:", {k: v for k, v in agent.summary().items()
                       if k in ("agent_type", "ai_vetoes", "ai_passes", "ai_upscales")})
    print("[OK] AI+ML agent smoke test passed.")
