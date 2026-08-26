"""
Tests for the FIFA/FC rating features in src.features:

  * squad_mean_rating — mean overall/potential of a squad's rated players at one
    tournament, but only if at least _MIN_RATED_PLAYERS (11) actually linked.
  * build_pool_index  — group the flat pool table into a {(team, version): [...]}
    lookup for fast per-match queries.
  * team_pool_rating  — a country's mean overall at a match's era-correct edition,
    with a leakage guard: only players already "known" by the match date count.

All pure (DataFrame/dict in, values out) — no DB needed.
"""

import math

import pandas as pd

from src.features import build_pool_index, squad_mean_rating, team_pool_rating


def _ratings_frame(overalls, potentials, team="Germany", tournament="Euro 2016"):
    """Small squad_ratings frame: one row per player, columns the feature reads."""
    return pd.DataFrame(
        {
            "team": team,
            "tournament_name": tournament,
            "overall": overalls,
            "potential": potentials,
        }
    )


class TestSquadMeanRating:
    def test_mean_over_enough_rated_players(self):
        # 11 rated players (the minimum) all at overall 80 / potential 85
        df = _ratings_frame([80] * 11, [85] * 11)
        out = squad_mean_rating(df, "Germany", "Euro 2016")
        assert out["n_rated"] == 11
        assert out["mean_overall"] == 80.0
        assert out["mean_potential"] == 85.0

    def test_too_few_rated_players_returns_nan(self):
        # only 10 linked ratings → below the 11-player floor → NaN (a shaky mean
        # dominated by whoever happened to link is worse than an honest "unknown")
        df = _ratings_frame([80] * 10, [85] * 10)
        out = squad_mean_rating(df, "Germany", "Euro 2016")
        assert out["n_rated"] == 10
        assert math.isnan(out["mean_overall"])
        assert math.isnan(out["mean_potential"])

    def test_unrated_rows_do_not_count_toward_the_floor(self):
        # 8 real ratings + 5 unlinked (NaN overall) = 13 rows but only 8 rated → NaN
        df = _ratings_frame([80] * 8 + [float("nan")] * 5, [85] * 13)
        out = squad_mean_rating(df, "Germany", "Euro 2016")
        assert out["n_rated"] == 8
        assert math.isnan(out["mean_overall"])


class TestBuildPoolIndex:
    def test_groups_by_team_and_version(self):
        pool = pd.DataFrame(
            {
                "team": ["Germany", "Germany", "Spain"],
                "sid": [1, 2, 3],
                "first_seen": pd.to_datetime(["2016-01-01", "2016-01-01", "2016-01-01"]),
                "version": [16, 16, 16],
                "overall": [80, 82, 79],
            }
        )
        index = build_pool_index(pool)
        assert set(index) == {("Germany", 16), ("Spain", 16)}
        assert len(index[("Germany", 16)]) == 2
        # each entry is a (first_seen, overall) tuple
        assert index[("Germany", 16)][0][1] in (80, 82)


class TestTeamPoolRating:
    def _pool_index(self, first_seens, overalls, team="X", version=22):
        pool = pd.DataFrame(
            {
                "team": team,
                "sid": range(len(overalls)),
                "first_seen": pd.to_datetime(first_seens),
                "version": version,
                "overall": overalls,
            }
        )
        return build_pool_index(pool)

    def test_mean_over_eligible_players(self):
        # 11 players all known before the 2022-06-01 match (edition 22) at overall 80
        idx = self._pool_index(["2020-01-01"] * 11, [80] * 11)
        rating = team_pool_rating(idx, "X", pd.Timestamp("2022-06-01"))
        assert rating == 80.0

    def test_leakage_guard_excludes_players_first_seen_after_the_match(self):
        # 10 known before the match + 1 first seen AFTER it → only 10 eligible →
        # below the 11 floor → NaN (a match can't be credited a future player)
        idx = self._pool_index(["2020-01-01"] * 10 + ["2023-01-01"], [80] * 11)
        rating = team_pool_rating(idx, "X", pd.Timestamp("2022-06-01"))
        assert math.isnan(rating)

    def test_uncovered_team_returns_nan(self):
        idx = self._pool_index(["2020-01-01"] * 11, [80] * 11)
        assert math.isnan(team_pool_rating(idx, "NotInPool", pd.Timestamp("2022-06-01")))
