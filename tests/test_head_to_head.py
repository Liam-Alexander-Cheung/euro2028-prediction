"""
Tests for src.features.head_to_head_record — team_a's weighted win rate against
one specific opponent, across ALL past meetings (no trailing window, unlike
rolling_form, because two nations meet too rarely for a 10-year cut).

Same oracle trick as the rolling_form tests: meetings on one shared date and
tournament carry equal weight, so the weighted mean collapses to the plain
average of the outcomes, computable by hand.
"""

import math

import pandas as pd
import pytest

from src.features import head_to_head_record


def test_equal_weight_h2h_is_plain_mean(make_matches):
    """A vs B, same-day friendlies: A wins 2, draws 1 → (1+1+0.5)/3 = 0.8333."""
    matches = make_matches(
        [
            {"date": "2000-01-01", "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 0},   # A win
            {"date": "2000-01-01", "home_team": "B", "away_team": "A",
             "home_score": 0, "away_score": 1},   # A win (as away side)
            {"date": "2000-01-01", "home_team": "A", "away_team": "B",
             "home_score": 1, "away_score": 1},   # draw
        ]
    )
    assert head_to_head_record(matches, "A", "B", pd.Timestamp("2001-01-01")) == pytest.approx(
        (1 + 1 + 0.5) / 3
    )


def test_perspective_is_symmetric_complement(make_matches):
    """From B's side the same fixtures give 1 - A's rate (draws count 0.5 for both)."""
    matches = make_matches(
        [
            {"date": "2000-01-01", "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 0},
            {"date": "2000-01-01", "home_team": "B", "away_team": "A",
             "home_score": 0, "away_score": 1},
            {"date": "2000-01-01", "home_team": "A", "away_team": "B",
             "home_score": 1, "away_score": 1},
        ]
    )
    a_rate = head_to_head_record(matches, "A", "B", pd.Timestamp("2001-01-01"))
    b_rate = head_to_head_record(matches, "B", "A", pd.Timestamp("2001-01-01"))
    assert b_rate == pytest.approx(1 - a_rate)


def test_other_teams_matches_are_ignored(make_matches):
    """Only A-vs-B meetings count; A's games against C must not leak in."""
    matches = make_matches(
        [
            {"date": "2000-01-01", "home_team": "A", "away_team": "B",
             "home_score": 3, "away_score": 0},   # the only A-vs-B game → A win → 1.0
            {"date": "2000-06-01", "home_team": "A", "away_team": "C",
             "home_score": 0, "away_score": 5},   # A loses to C — irrelevant here
        ]
    )
    assert head_to_head_record(matches, "A", "B", pd.Timestamp("2001-01-01")) == pytest.approx(1.0)


def test_uses_full_history_no_window(make_matches):
    """A meeting 20 years before the reference date still counts (no window cut),
    which is the whole reason h2h differs from rolling_form."""
    matches = make_matches(
        [
            {"date": "2000-01-01", "home_team": "A", "away_team": "B",
             "home_score": 1, "away_score": 0},   # 20 years before as_of → still counts
        ]
    )
    assert head_to_head_record(matches, "A", "B", pd.Timestamp("2020-01-01")) == pytest.approx(1.0)


def test_never_met_returns_nan(make_matches):
    """Two teams that have never met → NaN, never a 0.5 default."""
    matches = make_matches(
        [
            {"date": "2000-01-01", "home_team": "A", "away_team": "C",
             "home_score": 1, "away_score": 0},
        ]
    )
    assert math.isnan(head_to_head_record(matches, "A", "B", pd.Timestamp("2020-01-01")))
