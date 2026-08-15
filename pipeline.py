#!/usr/bin/env python3
"""
Shared pipeline for the Quantitative Sports Betting Model.

This module is the single source of truth for:

1. **Data generation** — a synthetic-but-realistic football world: each team has
   a latent strength, goals follow a Poisson process, and bookmaker odds are
   derived from the *true* outcome probabilities plus a margin and small
   mispricing noise.  (The original code sampled odds uniformly at random,
   which produced fake average edges of ~67% and absurd ROIs.)
2. **Model training** — PoissonElo (+ optional Gradient Boosting ML layer),
   and an optional Q-Learning staking agent trained on a *validation* split.
3. **Backtesting** — a train / validation / test backtest that converts model
   probabilities into edges, stakes via Kelly or the RL agent, resolves bets
   against the recorded results, and computes honest metrics.
4. **Metrics & plots** — ROI, strike rate, Sharpe, max drawdown, CLV, equity
   curve, edge / CLV distributions, P/L bars.

All randomness is seeded, so every run is reproducible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe: never open a GUI window
import matplotlib.pyplot as plt
import seaborn as sns

from models.poisson_elo_model import PoissonEloModel
from models.ml_layer import MLFootballPredictor
from models.rl_staking_agent import QLearningStakingAgent

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
TEAMS = [
    "Arsenal", "Man City", "Liverpool", "Chelsea", "Man Utd",
    "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "historical_matches.csv"
BACKTEST_DIR = PROJECT_ROOT / "backtests" / "results"

OUTCOMES = ["home_win", "draw", "away_win"]
RESULT_MAP = {"H": "home_win", "D": "draw", "A": "away_win"}

# Bookmaker world parameters (shared by the data generator and the demo
# simulation so both simulate the same world)
BOOKIE_GAMMA = 0.88          # favourite-longshot bias strength (gamma < 1)
BOOKIE_PROB_NOISE = 0.006    # per-outcome pricing noise (std, probability units)
BOOKIE_MARGIN_RANGE = (0.05, 0.08)  # overround applied to fair odds

# Default backtest parameters
INITIAL_BANKROLL = 1000.0
EDGE_THRESHOLD = 0.03       # only bet when edge > 3%
MIN_ODDS = 1.6              # skip very short prices
MIN_MODEL_PROB = 0.40       # only bet outcomes the model is confident about
                            # (longshot probabilities carry too much relative
                            # estimation error for reliable edge detection)
KELLY_FRACTION = 0.25       # quarter Kelly
MAX_STAKE_FRAC = 0.08       # hard cap on stake as % of bankroll
MIN_STAKE = 5.0             # ignore dust bets
SPLIT = (0.65, 0.15, 0.20)  # train / validation / test


# ----------------------------------------------------------------------------
# 1. Data generation
# ----------------------------------------------------------------------------
def _true_probs(lam_home: float, lam_away: float, max_goals: int = 10) -> Dict[str, float]:
    """True outcome probabilities from a Poisson goal model (the "world")."""
    h = [math.exp(-lam_home + k * math.log(lam_home) - math.lgamma(k + 1)) for k in range(max_goals + 1)]
    a = [math.exp(-lam_away + k * math.log(lam_away) - math.lgamma(k + 1)) for k in range(max_goals + 1)]
    p_home = p_draw = p_away = 0.0
    for i, ph in enumerate(h):
        for j, pa in enumerate(a):
            if i > j:
                p_home += ph * pa
            elif i == j:
                p_draw += ph * pa
            else:
                p_away += ph * pa
    total = p_home + p_draw + p_away
    return {"home_win": p_home / total, "draw": p_draw / total, "away_win": p_away / total}


def _make_bookie_odds(p_true: Dict[str, float], rng: np.random.Generator,
                      margin: float, prob_noise: float,
                      gamma: float = 0.94) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Opening and closing odds.

    The bookmaker prices from a *shaded* version of the true probabilities:
    p_bookie ~ normalize(p_true ** gamma) with gamma < 1, which reproduces the
    well-documented favorite-longshot bias (longshots priced too short,
    favourites priced too long).  A model that is as accurate as the bookmaker
    therefore finds persistent, realistic value on favourites.  A small pricing
    noise (prob_noise) and an overround (margin) are added on top.

    Closing odds use the same margin but an independent, smaller noise, so
    CLV = (closing - taken) / taken is centred near zero with a realistic
    spread instead of being systematically negative.
    """
    def _to_odds(p: Dict[str, float], m: float, s: float, g: float) -> Dict[str, float]:
        raw = np.array([p[k] for k in OUTCOMES]) ** g
        probs = np.clip(raw / raw.sum() + rng.normal(0, s, 3), 1e-4, 0.99)
        probs = probs / probs.sum()
        return {k: round(float((1.0 / probs[i]) * (1.0 + m)), 2) for i, k in enumerate(OUTCOMES)}

    opening = _to_odds(p_true, margin, prob_noise, gamma)
    closing = _to_odds(p_true, margin, prob_noise * 0.4, gamma)
    return opening, closing


def generate_match_data(n_matches: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic football world with realistic bookmaker odds.

    Returns a DataFrame with columns: date, home_team, away_team, home_goals,
    away_goals, result, odds_home_b365, odds_draw_b365, odds_away_b365,
    closing_odds_home, closing_odds_draw, closing_odds_away, margin.
    """
    rng = np.random.default_rng(seed)
    strengths = {t: float(rng.normal(0, 1)) for t in TEAMS}

    dates = pd.date_range("2022-08-01", periods=n_matches, freq="D")
    home_teams = rng.choice(TEAMS, n_matches)
    away_teams = []
    for h in home_teams:
        others = [t for t in TEAMS if t != h]
        away_teams.append(str(rng.choice(others)))
    away_teams = np.array(away_teams)

    diffs = np.array([strengths[h] - strengths[a] for h, a in zip(home_teams, away_teams)])
    lam_home = 1.6 * np.exp(0.22 * diffs) * 1.12  # home advantage
    lam_away = 1.3 * np.exp(-0.22 * diffs)

    home_goals = rng.poisson(lam_home)
    away_goals = rng.poisson(lam_away)
    result = np.where(home_goals > away_goals, "H",
                      np.where(home_goals < away_goals, "A", "D"))

    margins = rng.uniform(*BOOKIE_MARGIN_RANGE, n_matches)
    open_odds, close_odds = [], []
    for i in range(n_matches):
        p_true = _true_probs(float(lam_home[i]), float(lam_away[i]))
        o, c = _make_bookie_odds(p_true, rng, margin=float(margins[i]),
                                 prob_noise=BOOKIE_PROB_NOISE, gamma=BOOKIE_GAMMA)
        open_odds.append(o)
        close_odds.append(c)

    df = pd.DataFrame({
        "date": dates,
        "home_team": home_teams,
        "away_team": away_teams,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "odds_home_b365": [o["home_win"] for o in open_odds],
        "odds_draw_b365": [o["draw"] for o in open_odds],
        "odds_away_b365": [o["away_win"] for o in open_odds],
        "closing_odds_home": [c["home_win"] for c in close_odds],
        "closing_odds_draw": [c["draw"] for c in close_odds],
        "closing_odds_away": [c["away_win"] for c in close_odds],
        "margin": margins,
    })
    return df


def load_or_generate_data(n_matches: int = 1200, seed: int = 42,
                          regenerate: bool = False) -> Tuple[pd.DataFrame, bool]:
    """Load the processed CSV if present, otherwise generate it.

    Returns (df, generated).  `generated=True` means the CSV was (re)written.
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    required = ["closing_odds_home", "closing_odds_draw", "closing_odds_away"]
    if DATA_PATH.exists() and not regenerate:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        if all(c in df.columns for c in required):
            print(f"Loaded {len(df)} matches from {DATA_PATH.name}")
            return df, False
        print(f"{DATA_PATH.name} has an outdated schema - regenerating.")
    df = generate_match_data(n_matches=n_matches, seed=seed)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df)} matches -> {DATA_PATH}")
    return df, True


# ----------------------------------------------------------------------------
# 2. Model training
# ----------------------------------------------------------------------------
def train_models(train_df: pd.DataFrame, use_ml: bool = True,
                 verbose: bool = True) -> Tuple[PoissonEloModel, Optional[MLFootballPredictor]]:
    """Train the PoissonElo model (and optionally the ML layer).

    The ML layer is trained on the Elo-enriched training features so both
    models see identical, leakage-free features.
    """
    if verbose:
        print("  Training PoissonEloModel (Poisson regression + Elo)...")
    poisson = PoissonEloModel()
    train_feat = poisson.prepare_features(train_df)
    poisson.train(train_df)
    if verbose:
        print(f"  Learned Elo for {len(poisson.elo_ratings)} teams: "
              f"{min(poisson.elo_ratings.values()):.0f} - {max(poisson.elo_ratings.values()):.0f}")

    ml = None
    if use_ml:
        if verbose:
            print("  Training ML layer (Gradient Boosting)...")
        ml = MLFootballPredictor(model_type="gradient_boosting")
        metrics = ml.train(train_feat, verbose=verbose)
        if verbose:
            print(f"  ML validation: accuracy={metrics['accuracy']:.3f}, "
                  f"log-loss={metrics['log_loss']:.3f}")

    return poisson, ml


ENSEMBLE_SHRINK = 1.0   # pull ensemble probabilities toward base rates (1.0 = off;
                        # the PoissonElo model already applies its own shrinkage)


def ensemble_probs(poisson: PoissonEloModel, ml: Optional[MLFootballPredictor],
                   home: str, away: str) -> Dict[str, float]:
    """Blend PoissonElo and ML probabilities (average), then shrink toward the
    league base rates.

    Shrinkage matters: the value-betting subset is exactly where the model
    disagrees most with the bookmaker, so un-shrunk probabilities suffer from
    the *winner's curse* (extreme estimates win the selection and lose more
    often than claimed).  Shrinking the deviation from the base rates keeps the
    estimated edges honest and is a standard regularisation technique.
    """
    p_poisson = poisson.predict(home, away)
    if ml is None:
        blended = p_poisson
    else:
        p_ml = ml.predict_proba(home, away,
                                home_elo=poisson.get_team_elo(home),
                                away_elo=poisson.get_team_elo(away))
        blended = {k: (p_poisson[k] + p_ml[k]) / 2 for k in OUTCOMES}

    base = getattr(poisson, "base_rates", None)
    if base is not None and 0 < ENSEMBLE_SHRINK < 1.0:
        blended = {k: base[k] + ENSEMBLE_SHRINK * (blended[k] - base[k]) for k in OUTCOMES}
    return {k: round(blended[k], 4) for k in OUTCOMES}


# ----------------------------------------------------------------------------
# 3. Backtest
# ----------------------------------------------------------------------------
def _fractional_kelly(edge: float, odds: float, fraction: float = KELLY_FRACTION) -> float:
    if edge <= 0 or odds <= 1:
        return 0.0
    return min((edge / (odds - 1)) * fraction, MAX_STAKE_FRAC)


def _predictions_over(df: pd.DataFrame, poisson: PoissonEloModel,
                      ml: Optional[MLFootballPredictor]) -> pd.DataFrame:
    """Score a whole dataframe with the trained model(s) for validation."""
    rows = []
    for _, row in df.iterrows():
        probs = ensemble_probs(poisson, ml, row["home_team"], row["away_team"])
        rows.append({k: probs[k] for k in OUTCOMES})
    out = df.copy()
    for k in OUTCOMES:
        out[f"p_{k}"] = [r[k] for r in rows]
    return out


def evaluate_probability_quality(scored: pd.DataFrame) -> Dict[str, float]:
    """Log-loss, Brier score and accuracy of the model on a scored split.

    Uses all matches (not just bets), so it measures genuine predictive quality.
    """
    # Probability columns are ordered [away_win, draw, home_win], so the class
    # index for result H (home win) is 2, for A (away win) is 0.
    y_true = scored["result"].map({"H": 2, "D": 1, "A": 0}).to_numpy()
    probs = scored[["p_away_win", "p_draw", "p_home_win"]].to_numpy()
    eps = 1e-9
    log_loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], eps, 1))))
    brier = float(np.mean(np.sum((probs - np.eye(3)[y_true]) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(probs, axis=1) == y_true))
    n = len(y_true)
    # 3-class baseline: always predicting the most common outcome
    base_acc = float(max(np.bincount(y_true, minlength=3)) / n)
    return {
        "log_loss": round(log_loss, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(base_acc, 4),
        "n_matches": n,
    }


def _discovery_experiences(valid_df: pd.DataFrame, poisson: PoissonEloModel,
                           ml: Optional[MLFootballPredictor],
                           initial_bankroll: float) -> List[Tuple[float, float, float, bool]]:
    """Kelly backtest on the validation split, collecting realized bets as
    (edge, bankroll_pct, odds, win) experiences for the RL agent."""
    experiences = []
    bankroll = initial_bankroll
    for _, row in valid_df.iterrows():
        probs = ensemble_probs(poisson, ml, row["home_team"], row["away_team"])
        bookie = {"home_win": row["odds_home_b365"], "draw": row["odds_draw_b365"],
                  "away_win": row["odds_away_b365"]}
        edges = poisson.calculate_edge(probs, bookie, threshold=EDGE_THRESHOLD)
        best = edges.get("best_value")
        if not best or edges.get("max_edge", 0) < EDGE_THRESHOLD:
            continue
        odds = bookie[best]
        if odds < MIN_ODDS or probs[best] < MIN_MODEL_PROB:
            continue
        edge = edges[best]
        stake_frac = _fractional_kelly(edge, odds)
        stake = bankroll * stake_frac
        if stake < MIN_STAKE:
            continue
        win = RESULT_MAP.get(row["result"]) == best
        experiences.append((edge, bankroll / initial_bankroll, odds, win))
        bankroll += stake * (odds - 1) if win else -stake
    return experiences


def run_backtest(
    df: pd.DataFrame,
    use_ml: bool = True,
    use_rl: bool = True,
    initial_bankroll: float = INITIAL_BANKROLL,
    edge_threshold: float = EDGE_THRESHOLD,
    min_odds: float = MIN_ODDS,
    min_model_prob: float = MIN_MODEL_PROB,
    min_stake: float = MIN_STAKE,
    split: Tuple[float, float, float] = SPLIT,
    seed: int = 42,
    tag: str = "",
    save_results: bool = True,
    out_dir: Path = BACKTEST_DIR,
    verbose: bool = True,
) -> dict:
    """Run the full train / validation / test backtest.

    Returns a dict with: summary (metrics), bets_df, equity, models, and
    evaluation metrics for the validation and test splits.
    """
    if verbose:
        print("=" * 78)
        print("BACKTEST PIPELINE")
        print("=" * 78)

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])
    train_df = df.iloc[:n_train].copy()
    valid_df = df.iloc[n_train:n_train + n_valid].copy()
    test_df = df.iloc[n_train + n_valid:].copy()

    if verbose:
        print(f"  Split: train={len(train_df)} validation={len(valid_df)} test={len(test_df)}")

    poisson, ml = train_models(train_df, use_ml=use_ml, verbose=verbose)

    # Validation: score the model + collect RL experiences (no test leakage).
    valid_scored = _predictions_over(valid_df, poisson, ml)
    valid_eval = evaluate_probability_quality(valid_scored)

    rl_agent = None
    if use_rl:
        experiences = _discovery_experiences(valid_df, poisson, ml, initial_bankroll)
        if verbose:
            print(f"  Discovery backtest on validation: {len(experiences)} realized bets")
        rl_agent = QLearningStakingAgent()
        rl_agent.train(experiences, episodes=200)

    # Test: full backtest loop.
    if verbose:
        print(f"\n  Running test backtest "
              f"({'PoissonElo + ML + RL staking' if use_rl and use_ml else 'PoissonElo + Kelly'})...")
    bets: List[dict] = []
    bankroll = initial_bankroll
    equity = [bankroll]

    for _, row in test_df.iterrows():
        probs = ensemble_probs(poisson, ml, row["home_team"], row["away_team"])
        bookie = {"home_win": row["odds_home_b365"], "draw": row["odds_draw_b365"],
                  "away_win": row["odds_away_b365"]}
        edges = poisson.calculate_edge(probs, bookie, threshold=edge_threshold)
        best = edges.get("best_value")
        if not best or edges.get("max_edge", 0) < edge_threshold:
            continue
        odds = bookie[best]
        if odds < min_odds or probs[best] < min_model_prob:
            continue
        edge = edges[best]

        if rl_agent is not None:
            stake_frac = rl_agent.get_stake_fraction(edge, odds, bankroll, initial_bankroll)
            if stake_frac <= 0:
                stake_frac = _fractional_kelly(edge, odds)  # fallback
        else:
            stake_frac = _fractional_kelly(edge, odds)

        stake = round(bankroll * stake_frac, 2)
        if stake < min_stake:
            continue

        win = RESULT_MAP.get(row["result"]) == best
        profit = round(stake * (odds - 1), 2) if win else -stake
        bankroll = round(bankroll + profit, 2)

        closing_odds = row[f"closing_odds_{best.split('_')[0]}"]
        clv = round((closing_odds - odds) / odds * 100, 2) if closing_odds > 0 else 0.0

        bets.append({
            "date": str(row["date"].date()),
            "match": f"{row['home_team']} vs {row['away_team']}",
            "market": best,
            "my_odds": odds,
            "closing_odds": closing_odds,
            "stake": stake,
            "edge_pct": round(edge * 100, 2),
            "model": "PoissonElo+ML" if use_ml else "PoissonElo",
            "staking": "RL+Q-Learning" if rl_agent is not None else "Fractional-Kelly",
            "bet_outcome": "Win" if win else "Lose",
            "profit_loss": profit,
            "clv_pct": clv,
            "running_bankroll": bankroll,
        })
        equity.append(bankroll)

    bets_df = pd.DataFrame(bets)
    test_scored = _predictions_over(test_df, poisson, ml)
    test_eval = evaluate_probability_quality(test_scored)

    if verbose:
        print(f"  Generated {len(bets_df)} value bets on the test split")

    summary = compute_metrics(bets_df, equity, initial_bankroll, df, use_ml=use_ml,
                              use_rl=rl_agent is not None)
    layers = ["PoissonElo"]
    if use_ml:
        layers.append("ML(GB)")
    if rl_agent is not None:
        layers.append("RL(Q-Learning)")
    summary["layers_used"] = " + ".join(layers)

    if save_results:
        save_outputs(bets_df, equity, summary, tag=tag, out_dir=out_dir)

    return {
        "summary": summary,
        "bets_df": bets_df,
        "equity": equity,
        "models": {"poisson": poisson, "ml": ml, "rl": rl_agent},
        "validation_eval": valid_eval,
        "test_eval": test_eval,
        "splits": (train_df, valid_df, test_df),
    }


def compute_metrics(bets_df: pd.DataFrame, equity: List[float],
                    initial_bankroll: float, df: pd.DataFrame,
                    use_ml: bool = True, use_rl: bool = True) -> Dict[str, float]:
    """Compute backtest metrics from the bets log and equity curve."""
    total_bets = len(bets_df)
    if total_bets == 0:
        return {
            "total_bets": 0, "wins": 0, "losses": 0, "strike_rate": 0.0,
            "total_profit": 0.0, "roi_pct": 0.0, "avg_edge_pct": 0.0,
            "avg_clv_pct": 0.0, "avg_odds": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0, "final_bankroll": initial_bankroll,
            "profit_factor": 0.0, "cagr_pct": 0.0, "n_bets_per_year": 0.0,
        }

    wins = int((bets_df["bet_outcome"] == "Win").sum())
    losses = total_bets - wins
    total_profit = float(bets_df["profit_loss"].sum())
    roi = total_profit / initial_bankroll * 100

    returns = np.diff(equity) / np.array(equity[:-1])
    # Annualise with the actual bet frequency rather than an arbitrary 252.
    span_days = max((df["date"].max() - df["date"].min()).days, 1)
    bets_per_year = total_bets / (span_days / 365.25)
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bets_per_year)) if len(returns) > 1 and np.std(returns) > 0 else 0.0

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    max_dd = float(abs(((equity_arr - peak) / peak).min()) * 100)

    gross_wins = float(bets_df.loc[bets_df["profit_loss"] > 0, "profit_loss"].sum())
    gross_losses = float(-bets_df.loc[bets_df["profit_loss"] < 0, "profit_loss"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    final_bankroll = float(equity[-1])
    cagr = (final_bankroll / initial_bankroll) ** (365.25 / span_days) - 1 if final_bankroll > 0 else -1.0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "strike_rate": round(wins / total_bets * 100, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi, 2),
        "avg_edge_pct": round(float(bets_df["edge_pct"].mean()), 2),
        "avg_clv_pct": round(float(bets_df["clv_pct"].mean()), 2),
        "avg_odds": round(float(bets_df["my_odds"].mean()), 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "final_bankroll": round(final_bankroll, 2),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else None,
        "cagr_pct": round(cagr * 100, 2),
        "n_bets_per_year": round(bets_per_year, 1),
    }


# ----------------------------------------------------------------------------
# 4. Outputs
# ----------------------------------------------------------------------------
def save_outputs(bets_df: pd.DataFrame, equity: List[float],
                 summary: Dict[str, float], tag: str = "", out_dir: Path = BACKTEST_DIR):
    """Save the bets log, metrics file and plots (tagged with ``tag``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    bets_df.to_csv(out_dir / f"backtest_bets_log{suffix}.csv", index=False)

    layers = summary.get("layers_used", "PoissonElo")
    with open(out_dir / f"metrics{suffix}.txt", "w") as f:
        f.write(f"BACKTEST METRICS ({layers})\n")
        f.write("=" * 52 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    _plot_results(bets_df, equity, summary, out_dir, suffix)
    if len(bets_df) > 0:
        print(f"  [OK] Saved {out_dir / ('backtest_bets_log' + suffix + '.csv')}")
        print(f"  [OK] Saved {out_dir / ('metrics' + suffix + '.txt')}")
        print(f"  [OK] Saved {out_dir / ('backtest_analysis' + suffix + '.png')}")
        print(f"  [OK] Saved {out_dir / ('backtest_summary' + suffix + '.png')}")


def _plot_results(bets_df: pd.DataFrame, equity: List[float],
                  summary: Dict[str, float], out_dir: Path, suffix: str = ""):
    if len(bets_df) == 0:
        print("  No bets - skipping plots.")
        return
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    axes[0, 0].plot(equity, color="#2E86AB", linewidth=2.5)
    axes[0, 0].axhline(y=equity[0], color="gray", linestyle="--", alpha=0.8)
    axes[0, 0].set_title("Equity Curve", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Number of Bets"); axes[0, 0].set_ylabel("Bankroll ($)")

    sns.histplot(bets_df["edge_pct"], bins=20, ax=axes[0, 1], color="#A23B72", kde=True)
    axes[0, 1].axvline(x=EDGE_THRESHOLD * 100, color="red", linestyle="--", linewidth=2)
    axes[0, 1].set_title("Betting Edge Distribution", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Edge (%)")

    sns.histplot(bets_df["clv_pct"], bins=20, ax=axes[1, 0], color="#F18F01", kde=True)
    axes[1, 0].axvline(x=0, color="black", linestyle="--", linewidth=2)
    axes[1, 0].set_title("Closing Line Value (CLV) Distribution", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("CLV (%)")

    colors = ["#06D6A0" if p > 0 else "#EF476F" for p in bets_df["profit_loss"]]
    axes[1, 1].bar(range(len(bets_df)), bets_df["profit_loss"], color=colors, alpha=0.75)
    axes[1, 1].axhline(y=0, color="black", linewidth=1.5)
    axes[1, 1].set_title("Profit / Loss per Bet", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Bet Number"); axes[1, 1].set_ylabel("P/L ($)")

    plt.tight_layout()
    plt.savefig(out_dir / f"backtest_analysis{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in summary.items() if v is not None]
    text = "BACKTEST SUMMARY\n" + "=" * 45 + "\n\n" + "\n".join(lines)
    ax.text(0.05, 0.95, text, fontsize=11, family="monospace", verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#2E86AB", linewidth=2))
    plt.savefig(out_dir / f"backtest_summary{suffix}.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)


# ----------------------------------------------------------------------------
# 5. High-level entry points
# ----------------------------------------------------------------------------
def run_full_pipeline(n_matches: int = 1200, seed: int = 42, use_ml: bool = True,
                      use_rl: bool = True, regenerate: bool = False,
                      tag: str = "ml_rl") -> dict:
    """One call: data -> train -> backtest -> save.  Returns the result dict."""
    df, _ = load_or_generate_data(n_matches=n_matches, seed=seed, regenerate=regenerate)
    result = run_backtest(df, use_ml=use_ml, use_rl=use_rl, seed=seed, tag=tag)
    return result


def format_summary(summary: Dict[str, float]) -> str:
    """Human-readable metrics block for CLIs and demos."""
    lines = []
    for k, v in summary.items():
        if v is None:
            continue
        label = k.replace("_", " ").title()
        lines.append(f"  {label:<22}: {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_full_pipeline()
    print("\n=== SUMMARY ===")
    print(format_summary(res["summary"]))
