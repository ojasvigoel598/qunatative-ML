# Forward Prediction & Live P&L Logging Workflow

## Purpose
Build a genuine forward-looking track record. **All predictions must be logged BEFORE kickoff**. Results recorded AFTER match. This creates auditable, non-fabricated evidence for portfolio.

## Tools
- `predictions/forward_log.csv` (main log)
- `logs/pnl_log.csv` (updated with outcomes)
- `logs/clv_pnl_template.csv` (template)

## Workflow (Daily Use)

### 1. Pre-Match (Before Kickoff)
```python
# Example: Use model to generate prediction
from models.poisson_elo_model import PoissonEloModel
import pandas as pd

model = PoissonEloModel()
# ... load trained model ...

probs = model.predict("England", "Germany")
fair_odds = model.probs_to_fair_odds(probs)

# Bookie odds (from The Odds API or manual)
bookie = {"home_win": 2.10, "draw": 3.50, "away_win": 4.20}

edges = model.calculate_edge(probs, bookie)

if edges.get('best_value'):
    print(f"VALUE BET: {edges['best_value']} @ {bookie[edges['best_value']]} | Edge: {edges[edges['best_value']]:.2%}")
    
    # Log immediately (timestamped)
    log_entry = {
        'timestamp_logged': pd.Timestamp.now(),
        'match': 'England vs Germany (World Cup 2026)',
        'market': edges['best_value'],
        'my_odds': bookie[edges['best_value']],
        'model_prob': probs[edges['best_value']],
        'edge': edges[edges['best_value']],
        'stake': 50,  # or Kelly calculated
        'status': 'PENDING'
    }
    # Append to forward_log.csv
```

### 2. Log to CSV (Manual or Script)
Use the template in `logs/clv_pnl_template.csv`:
- Fill: date, match, market, my_odds, stake, edge_pct
- Leave outcome/CLV blank until after match

### 3. Post-Match (After Result)
- Update `bet_outcome`, `profit_loss`
- Fetch closing odds (The Odds API or manual)
- Calculate CLV = ((closing_odds / my_odds) - 1) * 100
- Update running_bankroll
- Move to `logs/pnl_log.csv`

### 4. Daily/Weekly Review
- Plot equity curve from backtest script
- Calculate live ROI, Sharpe, max DD
- Update project_summary.md with live stats

## Example Forward Log Entry (Pre-Match)
```
date: 2026-06-20
match: England vs Germany (WC Group)
market: Home Win
my_odds: 2.15
closing_odds: [to be filled]
stake: 45
edge_pct: 7.2
bet_outcome: [Win/Lose]
profit_loss: [calculated]
clv_pct: [calculated]
running_bankroll: 1045
notes: Model λh=2.1, λa=1.3 | Strong home edge
```

## Automation Ideas (Future)
- `scripts/03_live_predictor.py`: Fetch fixtures from football-data.org → predict → suggest bets
- Telegram/Discord bot for alerts
- Google Sheets sync for live dashboard

**Golden Rule**: Never backfill or edit pre-match entries. Timestamp everything.

*This workflow ensures the project remains 100% legitimate and presentable.*
