"""
Tests for src.features.goal_trend — weighted average goals scored / conceded /
differential for a team, same trailing-window + weighting convention as
rolling_form. Returns a dict of three values (kept separate on purpose: a 3-3
side and a 0.5-0.5 side both net zero difference but are very different teams).

Equal-weight oracle again: same date + tournament → weights cancel → each value
is the plain mean of the per-match goals.
"""

import math

import pandas as pd
import pytest

from src.features import goal_trend


def test_equal_weight_goal_means(make_matches):
    """
    Germany, three same-day friendlies: scores 3,1,2 and concedes 0,1,4.
    scored mean = (3+1+2)/3 = 2.0 ; conceded mean = (0+1+4)/3 = 1.6667 ;
    differential = 2.0 - 1.6667.
    """
    matches = make_matches(
        [
            {"date": "2010-01-01", "home_team": "Germany", "away_team": "A",
             "home_score": 3, "away_score": 0},
            {"date": "2010-01-01", "home_team": "B", "away_team": "Germany",
             "home_score": 1, "away_score": 1},   # Germany away: scored 1, conceded 1
            {"date": "2010-01-01", "home_team": "Germany", "away_team": "C",
             "home_score": 2, "away_score": 4},
        ]
    )
    trend = goal_trend(matches, "Germany", pd.Timestamp("2011-01-01"))
    assert trend["goals_scored"] == pytest.approx(2.0)
    assert trend["goals_conceded"] == pytest.approx(5 / 3)
    assert trend["goal_differential"] == pytest.approx(2.0 - 5 / 3)


def test_home_away_orientation(make_matches):
    """Scored/conceded must follow the team, not the home/away column: a single
    away game where Germany wins 0-2 means scored=2, conceded=0."""
    matches = make_matches(
        [
            {"date": "2010-01-01", "home_team": "A", "away_team": "Germany",
             "home_score": 0, "away_score": 2},
        ]
    )
    trend = goal_trend(matches, "Germany", pd.Timestamp("2011-01-01"))
    assert trend["goals_scored"] == pytest.approx(2.0)
    assert trend["goals_conceded"] == pytest.approx(0.0)


def test_no_history_returns_nan_pair(make_matches):
    """No qualifying matches → both goal figures NaN (not 0)."""
    matches = make_matches(
        [
            {"date": "2010-01-01", "home_team": "A", "away_team": "B",
             "home_score": 1, "away_score": 0},
        ]
    )
    trend = goal_trend(matches, "Germany", pd.Timestamp("2011-01-01"))
    assert math.isnan(trend["goals_scored"])
    assert math.isnan(trend["goals_conceded"])
