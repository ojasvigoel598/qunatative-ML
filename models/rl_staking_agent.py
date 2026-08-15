#!/usr/bin/env python3
"""
Reinforcement Learning Layer: Q-Learning agent for bet sizing.

The agent learns *how aggressively to bet relative to the fractional-Kelly
baseline* as a function of the observed state (edge, bankroll fraction).

State  : (edge_bin, bankroll_pct_bin)
Action : one of 5 discrete Kelly multipliers  {0x, 0.5x, 1x, 1.5x, 2x}
Stake  : multiplier * quarter-Kelly(edge, odds)
Reward : fractional bankroll change of the realized bet

Why Kelly multipliers instead of absolute stake percentages?  A naive agent
with absolute stake levels (0-10% of bankroll) tends to over-stake: at a
realistic edge of ~10% and odds ~2.2, quarter-Kelly is only ~2% of bankroll,
so absolute levels of 8-10% are 4-5x too aggressive and blow up variance.
Restricting the agent to multipliers of the principled Kelly baseline keeps
stakes sane while still letting the agent learn from realized outcomes.

The agent is trained on *realized* bets from a discovery backtest on the
validation split (no test leakage).
"""

import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

# Kelly multipliers (0x = no bet, 1.5x = 50% more than the Kelly baseline)
KELLY_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5]
N_ACTIONS = len(KELLY_MULTIPLIERS)

KELLY_FRACTION = 0.25      # quarter Kelly
MAX_STAKE_FRAC = 0.05      # hard cap, as a fraction of bankroll


def _quarter_kelly(edge: float, odds: float) -> float:
    if edge <= 0 or odds <= 1:
        return 0.0
    return min((edge / (odds - 1)) * KELLY_FRACTION, MAX_STAKE_FRAC)


class QLearningStakingAgent:
    def __init__(self, learning_rate: float = 0.1, discount: float = 0.9,
                 epsilon: float = 0.15):
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.kelly_multipliers = KELLY_MULTIPLIERS
        self.is_trained = False
        self.rng = np.random.default_rng(42)

    # ------------------------------------------------------------- State
    def discretize_state(self, edge: float, bankroll_pct: float):
        edge_bin = min(int(edge * 20), 9)          # 0-9
        bank_bin = min(int(bankroll_pct * 10), 9)  # 0-9
        return (edge_bin, bank_bin)

    def choose_action(self, state) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------- Train
    def train(self, experiences, episodes: int = 300):
        """Train on realized bets.

        Args:
            experiences: iterable of tuples (edge, bankroll_pct, odds, win)
                where ``win`` is True/False and ``odds`` is the decimal odds
                taken.  Typically collected from a Kelly-based discovery
                backtest on the validation split (no test leakage).
        """
        print(f"Training RL Staking Agent (Q-Learning, {episodes} episodes)...")
        experiences = list(experiences)
        if not experiences:
            print("  No experiences to learn from - agent stays untrained.")
            return

        for _ in range(episodes):
            for edge, bankroll_pct, odds, win in experiences:
                if edge <= 0.01 or odds <= 1.0:
                    continue
                state = self.discretize_state(edge, bankroll_pct)
                action = self.choose_action(state)
                stake_frac = _quarter_kelly(edge, odds) * self.kelly_multipliers[action]

                # Reward: fractional bankroll change from the realized bet.
                if win:
                    reward = stake_frac * (odds - 1.0)
                else:
                    reward = -stake_frac

                next_state = self.discretize_state(edge, bankroll_pct)
                current_q = self.q_table[state][action]
                self.q_table[state][action] = (
                    current_q
                    + self.lr * (reward + self.discount * np.max(self.q_table[next_state]) - current_q)
                )

        self.is_trained = True
        n_positive = sum(1 for e in experiences if e[3])
        print(f"  Trained on {len(experiences)} realized bets "
              f"({n_positive} wins, {len(experiences) - n_positive} losses)")

    # ------------------------------------------------------------- Use
    def get_stake_fraction(self, edge: float, odds: float,
                           current_bankroll: float, initial_bankroll: float) -> float:
        """Recommended stake fraction for the current state."""
        if not self.is_trained or edge <= 0.01:
            return 0.0
        state = self.discretize_state(edge, current_bankroll / max(initial_bankroll, 1e-9))
        action = int(np.argmax(self.q_table[state]))
        return _quarter_kelly(edge, odds) * self.kelly_multipliers[action]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Synthetic discovery bets: edges 3-15%, realistic odds, ~55% win rate.
    exp = []
    for _ in range(300):
        edge = rng.uniform(0.03, 0.15)
        odds = 1.0 / (0.5 + edge) * 1.05
        win = rng.random() < 0.55
        exp.append((edge, 1.0, odds, win))
    agent = QLearningStakingAgent()
    agent.train(exp, episodes=150)
    stake = agent.get_stake_fraction(0.07, 2.2, 1000, 1000)
    print(f"Recommended stake for 7% edge @ 2.20 odds: {stake:.2%} of bankroll")
    assert 0.0 <= stake <= 0.10
    print("[OK] QLearningStakingAgent self-test passed.")
