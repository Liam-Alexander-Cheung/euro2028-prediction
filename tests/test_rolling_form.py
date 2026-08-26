"""
Tests for src.features.rolling_form — the weighted-win-rate form feature that
feeds the XGBoost W/D/L model.

`rolling_form` is a pure function: matches DataFrame in, one float out. That
lets us hand it a few matches whose outcomes we choose, work out the expected
weighted win rate on paper, and assert the function reproduces it — plus pin
down the three behaviours that are easy to break by accident: the NaN-not-0.5
"no history" contract, the strict `< as_of_date` cut, and the trailing-window
cut.

The oracle trick used throughout: if every match shares the same date AND the
same tournament, its recency weight and importance weight are identical, so the
weights cancel and the weighted mean collapses to the plain average of the
outcomes (win=1, draw=0.5, loss=0). That makes the expected number computable
by hand with no reference to the exact weighting constants.
"""

import math

import pandas as pd
import pytest

from src.features import rolling_form


def test_equal_weight_average_is_plain_mean(make_matches):
    """4 same-day friendlies, 2 W / 1 D / 1 L → (1+1+0.5+0)/4 = 0.625."""
    matches = make_matches(
        [
            # win as home side (2-1)
            {"date": "2020-01-01", "home_team": "Germany", "away_team": "A",
             "home_score": 2, "away_score": 1},
            # win as away side (away 3, home 1) — exercises the home/away flip
            {"date": "2020-01-01", "home_team": "B", "away_team": "Germany",
             "home_score": 1, "away_score": 3},
            # draw (1-1)
            {"date": "2020-01-01", "home_team": "Germany", "away_team": "C",
             "home_score": 1, "away_score": 1},
            # loss (0-2)
            {"date": "2020-01-01", "home_team": "Germany", "away_team": "D",
             "home_score": 0, "away_score": 2},
        ]
    )
    result = rolling_form(matches, "Germany", pd.Timestamp("2021-01-01"))
    assert result == pytest.approx(0.625)


def test_recent_win_outweighs_old_loss(make_matches):
    """
    A win one month before the reference date and a loss eight years before it
    (both inside the 10-year window, same tournament) should pull the rate
    clearly ABOVE 0.5 — the recent result dominates because recency weight
    decays with age. Bracketed below 1.0 because the old loss still counts a
    little.
    """
    matches = make_matches(
        [
            {"date": "2019-12-01", "home_team": "Germany", "away_team": "A",
             "home_score": 1, "away_score": 0},   # recent win
            {"date": "2012-01-01", "home_team": "Germany", "away_team": "B",
             "home_score": 0, "away_score": 1},   # old loss (still within window)
        ]
    )
    result = rolling_form(matches, "Germany", pd.Timestamp("2020-01-01"))
    assert 0.5 < result < 1.0


def test_no_history_returns_nan(make_matches):
    """A team with zero qualifying matches returns NaN, never a 0.5 default —
    'no data' and 'known-average form' must stay distinguishable."""
    matches = make_matches(
        [
            {"date": "2020-01-01", "home_team": "Germany", "away_team": "A",
             "home_score": 1, "away_score": 0},
        ]
    )
    result = rolling_form(matches, "Atlantis", pd.Timestamp("2021-01-01"))
    assert math.isnan(result)


def test_match_on_reference_date_is_excluded(make_matches):
    """
    The filter is strict `< as_of_date`. Germany has a win BEFORE the reference
    date and a loss exactly ON it; only the win must count, so the rate is 1.0.
    If the boundary loss leaked in, the result would be 0.5 — so 1.0 proves the
    strict cut.
    """
    matches = make_matches(
        [
            {"date": "2020-01-01", "home_team": "Germany", "away_team": "A",
             "home_score": 1, "away_score": 0},   # counts
            {"date": "2020-06-01", "home_team": "Germany", "away_team": "B",
             "home_score": 0, "away_score": 1},   # ON as_of_date → excluded
        ]
    )
    result = rolling_form(matches, "Germany", pd.Timestamp("2020-06-01"))
    assert result == pytest.approx(1.0)


def test_match_older_than_window_is_excluded(make_matches):
    """
    Matches older than `window_years` (default 10) drop out. A win inside the
    window and a loss ~15 years before the reference date → only the win counts
    → 1.0. Again, 1.0 (not 0.5) proves the old loss was excluded.
    """
    matches = make_matches(
        [
            {"date": "2015-01-01", "home_team": "Germany", "away_team": "A",
             "home_score": 1, "away_score": 0},   # inside 10-year window
            {"date": "2005-01-01", "home_team": "Germany", "away_team": "B",
             "home_score": 0, "away_score": 1},   # outside window → excluded
        ]
    )
    result = rolling_form(matches, "Germany", pd.Timestamp("2020-06-01"))
    assert result == pytest.approx(1.0)


@pytest.mark.needs_db
def test_germany_outforms_san_marino_on_real_data(real_matches):
    """
    The project's classic real-football sanity check, promoted from a printed
    eyeball to an assertion: on the actual match database, Germany's form is
    well above 0.5 and clearly above San Marino's (football's archetypal
    minnow). Auto-skips when data/statxi.db is absent.
    """
    as_of = pd.Timestamp("2024-06-01")

    germany = rolling_form(real_matches, "Germany", as_of)
    san_marino = rolling_form(real_matches, "San Marino", as_of)

    assert not math.isnan(germany)
    assert not math.isnan(san_marino)
    assert germany > 0.5          # a strong side wins most of what it plays
    assert germany > san_marino   # and comfortably out-forms the minnow
