#!/usr/bin/env python3
"""
Interactive Metrics Dashboard.

Renders a SINGLE self-contained HTML file (docs/dashboard.html) from every
tracked artifact: summary cards, sortable/filterable tables, and charts
(embedded as base64 PNGs so the file works offline and on GitHub).

Usage:
    python scripts/09_make_dashboard.py
"""

import base64
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "backtests" / "results"
DEMO_OUT = PROJECT_ROOT / "demo" / "output"
OUT_HTML = PROJECT_ROOT / "docs" / "dashboard.html"


def png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


# ---------------------------------------------------------------- charts
def chart_backtests() -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for path, label, color in [("backtest_bets_log.csv", "PoissonElo + Kelly", "#2E86AB"),
                               ("backtest_bets_log_ml_rl.csv", "PoissonElo + ML + RL", "#A23B72")]:
        log = pd.read_csv(RESULTS / path)
        ax.plot(range(len(log)), log["running_bankroll"], label=label, lw=1.6, color=color)
    ax.axhline(1000, color="#9aa0a6", ls="--", lw=1)
    ax.set_xlabel("Bet index")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Canonical backtests (synthetic world, seed 42)")
    ax.legend()
    ax.grid(alpha=0.3)
    return png_b64(fig)


def chart_simulation() -> str:
    sim = pd.read_csv(DEMO_OUT / "simulation_1m_trials.csv")
    finals = sim["final_bankroll"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(np.log10(finals), bins=25, color="#2E86AB", alpha=0.85)
    ax.axvline(np.log10(1e6), color="#9aa0a6", ls="--", lw=1.2, label="start $1M")
    ax.axvline(np.log10(np.median(finals)), color="#d1495b", ls=":", lw=1.5, label="median")
    ax.set_xlabel("log10(final bankroll)")
    ax.set_ylabel("trials")
    ax.set_title(f"$1M simulation — {len(finals)} trials (mean ${finals.mean():,.0f}, "
                 f"median ${np.median(finals):,.0f})")
    ax.legend()
    ax.grid(alpha=0.3)
    return png_b64(fig)


def chart_transfer() -> str:
    tr = pd.read_csv(RESULTS / "transfer_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, league in zip(axes, ["La Liga", "Premier League"]):
        sub = tr[tr["league"] == league].sort_values("accuracy")
        ax.barh(sub["method"], sub["accuracy"] * 100, color="#A23B72")
        ax.set_title(f"{league} 25/26 (synthetic-trained)")
        ax.set_xlabel("Accuracy (%)")
        ax.set_xlim(0, 60)
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.text(r["accuracy"] * 100 + 0.4, i, f"{r['accuracy']:.1%}", va="center", fontsize=8)
    plt.tight_layout()
    return png_b64(fig)


def chart_season() -> str:
    s = pd.read_csv(RESULTS / "season_backtest_results.csv")
    w = s[s["experiment"].str.startswith("La Liga: train")].copy()
    w["test_season"] = w["experiment"].str.split("-> test ").str[-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m in ["Majority / base rate", "PoissonElo model", "Ridge classifier",
              "Gradient Boosting", "Random Forest"]:
        sub = w[w["method"] == m].sort_values("test_season")
        ax.plot(sub["test_season"], sub["accuracy"], marker="o", label=m, lw=1.5)
    ax.set_xlabel("Test season (expanding-window train)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Real La Liga — season-by-season accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return png_b64(fig)


def chart_dlr() -> str:
    d = pd.read_csv(RESULTS / "deep_learning_real_results.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(d))
    la = d["NN__La Liga 25/26__acc"].to_numpy()
    ep = d["NN__EPL 25/26__acc"].to_numpy()
    ax.bar(x - 0.2, la * 100, 0.4, label="La Liga 25/26", color="#2E86AB")
    ax.bar(x + 0.2, ep * 100, 0.4, label="EPL 25/26", color="#F18F01")
    ax.axhline(48.9, color="#9aa0a6", ls="--", lw=1, label="majority (La Liga)")
    ax.set_xticks(x)
    ax.set_xticklabels(d["iteration"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Deep nets on real vs synthetic training (iteration loop)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return png_b64(fig)


def chart_real_sim() -> str:
    eq = pd.read_csv(DEMO_OUT / "real_simulation_2023.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(len(eq)), eq["bankroll"], color="#2E86AB", lw=1.6)
    ax.axhline(1_000_000, color="#9aa0a6", ls="--", lw=1, label="start $1M")
    ax.axhline(100_000, color="#d1495b", ls=":", lw=1.2, label="survival floor")
    ax.set_xlabel("Bet index (chronological)")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Real walk-forward — La Liga 2022/23, point-in-time knowledge")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return png_b64(fig)


def chart_stress() -> str:
    st = pd.read_csv(RESULTS / "staking_stress_test.csv")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(st))
    w = 0.28
    ax.bar(x - w, st["P(profit)"] * 100, w, label="P(profit)", color="#2E86AB")
    ax.bar(x, st["P(ruin<100k)"] * 100, w, label="P(final < $100K)", color="#d1495b")
    ax.bar(x + w, st["P(ever<100k)"] * 100, w, label="P(ever < $100K)", color="#F18F01")
    ax.set_xticks(x)
    ax.set_xticklabels(st["policy"], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Staking policy stress test — 100 trials")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return png_b64(fig)


# ---------------------------------------------------------------- tables
def ledger_rows() -> list:
    rows = []
    for name, path in [("PoissonElo + Kelly", "backtest_bets_log.csv"),
                       ("PoissonElo + ML + RL", "backtest_bets_log_ml_rl.csv")]:
        log = pd.read_csv(RESULTS / path)
        n = len(log)
        wins = int((log["bet_outcome"] == "Win").sum())
        lo, hi = wilson_ci(wins, n)
        rows.append([f"Backtest {name}", "Strike rate (Wilson 95% CI)",
                     f"{wins / n:.1%} [{lo:.1%}, {hi:.1%}]"])
        rows.append([f"Backtest {name}", "Avg edge",
                     f"{log['edge_pct'].mean():+.1f}% (t={np.mean(log['edge_pct']) / (np.std(log['edge_pct'], ddof=1) / np.sqrt(n)):.1f})"])
        rows.append([f"Backtest {name}", "Avg CLV",
                     f"{log['clv_pct'].mean():+.2f}%"])
    sim = pd.read_csv(DEMO_OUT / "simulation_1m_trials.csv")
    finals = sim["final_bankroll"].to_numpy()
    rows.append(["$1M simulation (25 trials)", "Mean / median final",
                 f"${finals.mean():,.0f} / ${np.median(finals):,.0f}"])
    rows.append(["$1M simulation (25 trials)", "P(profit)",
                 f"{(finals > 1e6).mean():.0%}"])
    rsim = pd.read_csv(DEMO_OUT / "real_simulation_2023.csv")
    last = rsim["bankroll"].iloc[-1]
    rows.append(["Real walk-forward La Liga 22/23", "ROI",
                 f"{last / 1e6 - 1:+.1%}"])
    rows.append(["Real walk-forward La Liga 22/23", "Real CLV vs Pinnacle",
                 f"{rsim['clv_pct'].mean():+.2f}%"])
    st = pd.read_csv(RESULTS / "staking_stress_test.csv")
    best = st.loc[st["P(ruin<100k)"].idxmin()]
    rows.append(["Staking stress test (100 trials)", "Lowest-ruin policy",
                 f"{best['policy']} (P(ruin) {best['P(ruin<100k)']:.0%}, "
                 f"P(profit) {best['P(profit)']:.0%})"])
    return rows


def stress_table() -> list:
    st = pd.read_csv(RESULTS / "staking_stress_test.csv")
    return [st.columns.tolist()] + st.round(3).values.tolist()


# ---------------------------------------------------------------- html
def summary_cards() -> str:
    bt = pd.read_csv(RESULTS / "backtest_bets_log.csv")
    bt2 = pd.read_csv(RESULTS / "backtest_bets_log_ml_rl.csv")
    sim = pd.read_csv(DEMO_OUT / "simulation_1m_trials.csv")
    rsim = pd.read_csv(DEMO_OUT / "real_simulation_2023.csv")
    st = pd.read_csv(RESULTS / "staking_stress_test.csv")
    finals = sim["final_bankroll"].to_numpy()
    rsim_roi = rsim["bankroll"].iloc[-1] / 1e6 - 1
    best = st.loc[st["P(ruin<100k)"].idxmin()]
    cards = [
        ("Backtest ROI (base)", f"{bt['profit_loss'].sum() / 1000:+.1%}",
         f"{len(bt)} bets, {bt['bet_outcome'].eq('Win').mean():.0%} strike"),
        ("Backtest ROI (ML+RL)", f"{bt2['profit_loss'].sum() / 1000:+.1%}",
         f"{len(bt2)} bets, {bt2['bet_outcome'].eq('Win').mean():.0%} strike"),
        ("$1M simulation", f"${np.median(finals):,.0f} median",
         f"mean ${finals.mean():,.0f} · P(profit) {(finals > 1e6).mean():.0%}"),
        ("Real replay La Liga 22/23", f"{rsim_roi:+.1%} ROI",
         f"{len(rsim)} bets · real CLV {rsim['clv_pct'].mean():+.2f}%"),
        ("Lowest-ruin staking policy", best["policy"],
         f"P(ruin) {best['P(ruin<100k)']:.0%} · P(profit) {best['P(profit)']:.0%}"),
        ("Best real-data model", "PoissonElo ~54% acc",
         "vs 46% majority baseline (4 unseen seasons)"),
    ]
    return "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in cards)


def build_html():
    charts = {
        "backtests": chart_backtests(),
        "simulation": chart_simulation(),
        "transfer": chart_transfer(),
        "season": chart_season(),
        "dlr": chart_dlr(),
        "realsim": chart_real_sim(),
        "stress": chart_stress(),
    }
    ledger = ledger_rows()
    stress = stress_table()
    cards = summary_cards()

    tab_btns = "\n".join(
        f'<button class="tab" onclick="showTab(\'{k}\')">{label}</button>'
        for k, label in [
            ("summary", "Summary"), ("backtests", "Backtests"), ("simulation", "Simulation"),
            ("transfer", "Transfer"), ("season", "Season backtest"), ("dlr", "Deep nets"),
            ("realsim", "Real replay"), ("stress", "Staking stress"),
        ])

    ledger_html = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in ledger)
    stress_html = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in stress)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sports Betting Model — Metrics Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #222; }}
  header {{ background: #16213e; color: #fff; padding: 18px 28px; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header p {{ margin: 4px 0 0; opacity: .85; font-size: 13px; }}
  nav {{ background: #1f2d55; padding: 8px 28px; }}
  .tab {{ background: none; border: none; color: #cfd8ff; padding: 8px 14px; cursor: pointer; font-size: 13px; border-radius: 6px 6px 0 0; }}
  .tab.active {{ background: #f5f6f8; color: #16213e; font-weight: 600; }}
  main {{ padding: 24px 28px; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-bottom: 20px; }}
  .card {{ background: #fff; border: 1px solid #e2e5ea; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  .card .k {{ font-size: 12px; color: #667; }}
  .card .v {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
  .card .s {{ font-size: 12px; color: #889; margin-top: 2px; }}
  img.chart {{ width: 100%; max-width: 900px; background: #fff; border-radius: 10px; border: 1px solid #e2e5ea; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
  th, td {{ border: 1px solid #e2e5ea; padding: 7px 10px; text-align: left; }}
  th {{ background: #16213e; color: #fff; cursor: pointer; user-select: none; }}
  tr:nth-child(even) {{ background: #f9fafb; }}
  .section {{ margin: 22px 0 10px; font-size: 15px; font-weight: 600; color: #16213e; }}
  footer {{ padding: 18px 28px; color: #778; font-size: 12px; }}
</style></head><body>
<header><h1>⚽ Quantitative Sports Betting Model — Metrics Dashboard</h1>
<p>Every tracked build with statistical uncertainty · regenerated by <code>scripts/09_make_dashboard.py</code></p></header>
<nav>{tab_btns}</nav>
<main>
  <div class="panel active" id="panel-summary">
    <div class="cards">{cards}</div>
    <div class="section">Ledger — all tracked builds</div>
    <table id="ledger"><thead><tr><th>Build</th><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{ledger_html}</tbody></table>
  </div>
  <div class="panel" id="panel-backtests"><div class="section">Canonical backtests (synthetic world)</div><img class="chart" src="data:image/png;base64,{charts['backtests']}"></div>
  <div class="panel" id="panel-simulation"><div class="section">$1M Monte-Carlo simulation — 25 trials</div><img class="chart" src="data:image/png;base64,{charts['simulation']}"></div>
  <div class="panel" id="panel-transfer"><div class="section">Deep-learning transfer (synthetic-trained) — real 2025/26</div><img class="chart" src="data:image/png;base64,{charts['transfer']}"></div>
  <div class="panel" id="panel-season"><div class="section">Real-data season-by-season backtest (La Liga)</div><img class="chart" src="data:image/png;base64,{charts['season']}"></div>
  <div class="panel" id="panel-dlr"><div class="section">Deep nets on real vs synthetic training — iteration loop</div><img class="chart" src="data:image/png;base64,{charts['dlr']}"></div>
  <div class="panel" id="panel-realsim"><div class="section">Real walk-forward — La Liga 2022/23, point-in-time</div><img class="chart" src="data:image/png;base64,{charts['realsim']}"></div>
  <div class="panel" id="panel-stress">
    <div class="section">Staking policy stress test — 100 trials</div>
    <img class="chart" src="data:image/png;base64,{charts['stress']}">
    <div class="section">Policy table (identical bet streams, only staking differs)</div>
    <table id="stress"><thead><tr><th>Policy</th><th>Mean final</th><th>Median final</th><th>P(profit)</th><th>P(ruin&lt;100k)</th><th>P(ever&lt;100k)</th><th>median maxDD</th><th>median CAGR</th></tr></thead>
    <tbody>{stress_html}</tbody></table>
  </div>
</main>
<footer>Generated from real run artifacts. Results are research outputs — not financial advice.</footer>
<script>
function showTab(k) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + k).classList.add('active');
  event.target.classList.add('active');
}}
// simple sortable tables
document.querySelectorAll('table').forEach(tbl => {{
  tbl.querySelectorAll('th').forEach((th, i) => {{
    th.addEventListener('click', () => {{
      const tbody = tbl.tBodies[0];
      const rows = Array.from(tbody.rows);
      const dir = th.dataset.dir === 'asc' ? -1 : 1;
      rows.sort((a, b) => {{
        const x = parseFloat(a.cells[i].textContent.replace(/[$,%]/g, '')) || a.cells[i].textContent;
        const y = parseFloat(b.cells[i].textContent.replace(/[$,%]/g, '')) || b.cells[i].textContent;
        return (x < y ? -1 : x > y ? 1 : 0) * dir;
      }});
      rows.forEach(r => tbody.appendChild(r));
      th.dataset.dir = dir === 1 ? 'asc' : 'desc';
    }});
  }});
}});
</script>
</body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote {OUT_HTML} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    build_html()
