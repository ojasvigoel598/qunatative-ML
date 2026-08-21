#!/usr/bin/env python3
"""
Betting Tracker — web-based UI matching the desktop application.

Tabs: Pending Bets, All Bets, Equity Curve, League Trust, Daily P/L,
      CLV Analysis, Analytics, Advisor, Odds Movement

Sidebar: Overall Stats, Per-League Stats, Daily P/L

Usage:
    python tracker/app.py
    # Open http://localhost:5000 in browser
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))

RESULTS_DIR = PROJECT_ROOT / "backtests" / "results"
DATA_DIR = PROJECT_ROOT / "data" / "real"


def load_bets_data():
    """Load all available bet data from CSV files."""
    all_bets = []

    # Try synthetic backtest bets
    csv_path = RESULTS_DIR / "backtest_bets_log.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            parts = row.get("match", " vs ").split(" vs ")
            home = parts[0].strip() if len(parts) > 0 else ""
            away = parts[1].strip() if len(parts) > 1 else ""
            all_bets.append({
                "date": str(row.get("date", "")),
                "home": home,
                "away": away,
                "prediction": row.get("market", ""),
                "odds": float(row.get("my_odds", 0)),
                "prob": 0.0,
                "league": "Synthetic",
                "edge": float(row.get("edge_pct", 0)),
                "stake": float(row.get("stake", 0)),
                "result": row.get("bet_outcome", ""),
                "pnl": float(row.get("profit_loss", 0)),
                "balance": float(row.get("running_bankroll", 0)),
                "closing": float(row.get("closing_odds", 0)),
                "clv": float(row.get("clv_pct", 0)),
            })

    # Try CLV-focused results
    csv_path = RESULTS_DIR / "clv_focused_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            all_bets.append({
                "date": str(row.get("date", "")),
                "home": "",
                "away": "",
                "prediction": "",
                "odds": float(row.get("avg_odds", 0)),
                "prob": 0.0,
                "league": "CLV-Focused",
                "edge": float(row.get("avg_edge_pct", 0)),
                "stake": float(row.get("avg_stake", 0)),
                "result": "Win" if row.get("roi_pct", 0) > 0 else "Lose",
                "pnl": float(row.get("total_profit", 0)),
                "balance": float(row.get("final_bankroll", 10000)),
                "closing": 0,
                "clv": float(row.get("clv_mean_pct", 0)),
            })

    # Try real data backtest results
    csv_path = RESULTS_DIR / "clv_real_data_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            all_bets.append({
                "date": str(row.get("test_season", "")),
                "home": "",
                "away": "",
                "prediction": row.get("strategy", ""),
                "odds": float(row.get("avg_odds", 0)) if "avg_odds" in row else 0,
                "prob": 0.0,
                "league": "Real Data",
                "edge": float(row.get("avg_edge_pct", 0)) if "avg_edge_pct" in row else 0,
                "stake": float(row.get("avg_stake", 0)) if "avg_stake" in row else 0,
                "result": "Win" if row.get("roi_pct", 0) > 0 else "Lose",
                "pnl": float(row.get("total_profit", 0)) if "total_profit" in row else 0,
                "balance": float(row.get("final_bankroll", 10000)),
                "closing": 0,
                "clv": float(row.get("clv_mean_pct", 0)),
            })

    return all_bets


def compute_stats(bets):
    """Compute overall statistics from bet data."""
    if not bets:
        return {
            "starting_balance": 10000,
            "current_balance": 10000,
            "total_bets": 0,
            "wins": 0,
            "win_rate": 0,
            "avg_odds": 0,
            "profit": 0,
            "yield_pct": 0,
            "bankroll_growth": 0,
        }

    total = len(bets)
    wins = sum(1 for b in bets if b["result"] == "Win")
    total_stake = sum(b["stake"] for b in bets)
    total_profit = sum(b["pnl"] for b in bets)
    avg_odds = np.mean([b["odds"] for b in bets if b["odds"] > 0]) if any(b["odds"] > 0 for b in bets) else 0
    starting = bets[0]["balance"] - bets[0]["pnl"] if bets else 10000
    current = bets[-1]["balance"] if bets else 10000

    return {
        "starting_balance": round(starting, 2),
        "current_balance": round(current, 2),
        "total_bets": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "avg_odds": round(float(avg_odds), 2),
        "profit": round(total_profit, 2),
        "yield_pct": round(total_profit / total_stake * 100, 1) if total_stake > 0 else 0,
        "bankroll_growth": round((current - starting) / starting * 100, 2) if starting > 0 else 0,
    }


def compute_league_stats(bets):
    """Compute per-league statistics."""
    leagues = {}
    for b in bets:
        lg = b.get("league", "Unknown")
        if lg not in leagues:
            leagues[lg] = {"bets": 0, "wins": 0, "profit": 0, "stake": 0}
        leagues[lg]["bets"] += 1
        if b["result"] == "Win":
            leagues[lg]["wins"] += 1
        leagues[lg]["profit"] += b["pnl"]
        leagues[lg]["stake"] += b["stake"]

    result = []
    for lg, stats in leagues.items():
        result.append({
            "league": lg,
            "bets": stats["bets"],
            "wins": stats["wins"],
            "win_rate": round(stats["wins"] / stats["bets"] * 100, 1) if stats["bets"] > 0 else 0,
            "profit": round(stats["profit"], 2),
            "yield": round(stats["profit"] / stats["stake"] * 100, 1) if stats["stake"] > 0 else 0,
        })
    return sorted(result, key=lambda x: x["profit"], reverse=True)


def compute_daily_pnl(bets):
    """Compute daily P/L breakdown."""
    daily = {}
    cumulative = 0
    for b in bets:
        date = b["date"][:10] if len(b["date"]) >= 10 else b["date"]
        if date not in daily:
            daily[date] = {"date": date, "daily_pnl": 0, "bets": 0, "wins": 0}
        daily[date]["daily_pnl"] += b["pnl"]
        daily[date]["bets"] += 1
        if b["result"] == "Win":
            daily[date]["wins"] += 1

    result = []
    for date in sorted(daily.keys()):
        cumulative += daily[date]["daily_pnl"]
        result.append({
            "date": date,
            "daily_pnl": round(daily[date]["daily_pnl"], 2),
            "cumulative": round(cumulative, 2),
            "bets": daily[date]["bets"],
            "wins": daily[date]["wins"],
        })
    return result


def compute_clv_analysis(bets):
    """Compute CLV analysis."""
    clv_bets = [b for b in bets if b["clv"] != 0]
    if not clv_bets:
        return {"sufficient_data": False}

    pos_clv = [b for b in clv_bets if b["clv"] > 0]
    neg_clv = [b for b in clv_bets if b["clv"] <= 0]

    pos_wins = sum(1 for b in pos_clv if b["result"] == "Win")
    neg_wins = sum(1 for b in neg_clv if b["result"] == "Win")

    return {
        "sufficient_data": True,
        "total_clv_bets": len(clv_bets),
        "positive_clv_bets": len(pos_clv),
        "negative_clv_bets": len(neg_clv),
        "positive_clv_win_rate": round(pos_wins / len(pos_clv) * 100, 1) if pos_clv else 0,
        "negative_clv_win_rate": round(neg_wins / len(neg_clv) * 100, 1) if neg_clv else 0,
        "avg_clv": round(float(np.mean([b["clv"] for b in clv_bets])), 2),
        "clv_positive_rate": round(float(np.mean([b["clv"] > 0 for b in clv_bets])) * 100, 1),
    }


@app.route("/")
def index():
    """Main tracker page."""
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    """Return all bet data as JSON."""
    bets = load_bets_data()
    stats = compute_stats(bets)
    league_stats = compute_league_stats(bets)
    daily_pnl = compute_daily_pnl(bets)
    clv_analysis = compute_clv_analysis(bets)

    return jsonify({
        "bets": bets,
        "stats": stats,
        "league_stats": league_stats,
        "daily_pnl": daily_pnl,
        "clv_analysis": clv_analysis,
    })


@app.route("/api/equity")
def api_equity():
    """Return equity curve data."""
    bets = load_bets_data()
    equity = [{"balance": 10000, "index": 0}]
    for i, b in enumerate(bets):
        equity.append({"balance": b["balance"], "index": i + 1})
    return jsonify(equity)


if __name__ == "__main__":
    print("=" * 60)
    print("BETTING TRACKER")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    app.run(debug=True, port=5000)
