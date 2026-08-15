#!/usr/bin/env python3
"""
Randomised multi-league world for the chronological agent simulation.

Every run draws a fresh scenario from a stored seed:
  * a random subset of leagues (2-4 of La Liga / Premier League / Bundesliga
    / Serie A),
  * a random WALK season (only from COMPLETE seasons, so every result
    genuinely exists),
  * a random start date inside the walk season (so runs begin at different
    points of the calendar),
  * a random REVEAL ORDER for the leagues: one league is visible at the
    start, the others become visible at random later dates within the first
    part of the walk — the agent can only see fixtures of leagues whose
    reveal date has passed.

The world exposes, per simulation day, only what would be public at that
time:
  * fixtures of REVEALED leagues with kickoff within a lookahead window
    (the upcoming "matches available" — schedule is public knowledge),
  * results of matches that have already kicked off (revealed chronologically).

Nothing computed from future dates is ever attached to a match row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent_sim.fetch import (COMPLETE_SEASONS, LEAGUES, SEASON_CODES,
                             SEASON_LABEL, fetch_seasons)

POOL_LEAGUES = ["SP1", "E0", "D1", "I1"]   # La Liga, EPL, Bundesliga, Serie A


class World:
    """One reproducible, randomised simulation scenario."""

    def __init__(self, seed: int, leagues=None, walk_season: str = None,
                 start_frac=(0.04, 0.35), lookahead_days: int = 3,
                 min_leagues: int = 2, max_leagues: int = 4,
                 offline: bool = False):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.offline = offline
        self.lookahead_days = int(lookahead_days)

        # ---- random league subset
        n = int(self.rng.integers(min_leagues, max_leagues + 1))
        self.leagues = (list(leagues) if leagues
                        else list(self.rng.choice(POOL_LEAGUES, size=n,
                                                  replace=False)))
        self.league_names = {lg: LEAGUES[lg] for lg in self.leagues}

        # ---- random walk season (complete only) + prior training seasons
        self.walk_season = walk_season or str(self.rng.choice(COMPLETE_SEASONS))
        idx = SEASON_CODES.index(self.walk_season)
        self.prior_seasons = SEASON_CODES[:idx]
        self.train_df = fetch_seasons(self.leagues, self.prior_seasons,
                                      offline)
        self.walk_df = fetch_seasons(self.leagues, [self.walk_season],
                                     offline)

        # ---- random start date inside the walk season.  EVERYTHING before the
        # start date (prior seasons + early walk-season matches) is legitimate
        # training history for the agent; the walk is what comes after.
        d0, d1 = self.walk_df["date"].min(), self.walk_df["date"].max()
        span = max(int((d1 - d0).days), 1)
        frac = float(self.rng.uniform(*start_frac))
        self.start_date = d0 + pd.Timedelta(days=int(span * frac))
        pre_start = self.walk_df[self.walk_df["date"] < self.start_date]
        self.train_df = pd.concat([self.train_df, pre_start], ignore_index=True) \
            .sort_values("date").reset_index(drop=True)
        self.walk = self.walk_df[self.walk_df["date"] >= self.start_date] \
            .copy().reset_index(drop=True)
        self.end_date = self.walk_df["date"].max()
        # attach the football-data league code to every walk row (needed by
        # the reveal map and the ledger)
        self.walk["league_code"] = self.walk["league"].map(LEAGUE_CODE)

        # ---- random reveal schedule (first league visible at start)
        order = list(self.rng.permutation(self.leagues))
        offsets = sorted(self.rng.uniform(0.02, 0.30, size=len(self.leagues)))
        self.reveal_order = order
        self.reveal = {}
        for i, lg in enumerate(order):
            off_days = 0 if i == 0 else int(span * float(offsets[i - 1]))
            self.reveal[lg] = self.start_date + pd.Timedelta(days=off_days)

        # ---- walk dates (all unique dates in the walk window)
        self.sim_dates = sorted(
            pd.unique(self.walk[self.walk["date"] >= self.start_date]["date"]))
        self.matches_by_date = {d: [] for d in self.sim_dates}
        for _, r in self.walk.iterrows():
            self.matches_by_date.setdefault(r["date"], []).append(r)
        # index: kickoff date -> rows, for quick resolution lookups
        self.index = {d: self.matches_by_date[d] for d in self.sim_dates}

    # ------------------------------------------------------------ queries
    def revealed_leagues(self, sim_day: pd.Timestamp) -> list:
        return [lg for lg in self.leagues
                if self.reveal[lg] <= sim_day]

    def upcoming_matches(self, sim_day: pd.Timestamp) -> list:
        """Matches of revealed leagues kicking off in (sim_day, +lookahead].

        The fixture list (date + teams + pre-match odds) is public schedule
        knowledge; only the RESULT is withheld until kickoff.
        """
        horizon = sim_day + pd.Timedelta(days=self.lookahead_days)
        out = []
        for d in self.sim_dates:
            if sim_day < d <= horizon:
                for m in self.matches_by_date[d]:
                    if self.reveal[m["league_code"]] <= sim_day:
                        out.append(m)
        return out

    def results_on(self, sim_day: pd.Timestamp) -> list:
        """Matches that kick off exactly on sim_day (results become known)."""
        return list(self.matches_by_date.get(sim_day, []))

    # -------------------------------------------------------------- report
    def describe(self) -> dict:
        return {
            "seed": self.seed,
            "leagues": ", ".join(f"{lg} ({LEAGUES[lg]})" for lg in self.leagues),
            "walk_season": SEASON_LABEL[self.walk_season],
            "train_matches": int(len(self.train_df)),
            "walk_matches": int(len(self.walk)),
            "start_date": str(self.start_date.date()),
            "end_date": str(self.end_date.date()),
            "lookahead_days": self.lookahead_days,
            "reveal_order": " -> ".join(f"{lg}" for lg in self.reveal_order),
            "reveal_dates": ", ".join(
                f"{lg}:{self.reveal[lg].date()}" for lg in self.leagues),
        }


def _add_league_code(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the football-data code to each row (league name -> code)."""
    code = {v: k for k, v in LEAGUES.items()}
    df = df.copy()
    df["league_code"] = df["league"].map(code)
    return df


LEAGUE_CODE = {v: k for k, v in LEAGUES.items()}
