"""
Tests for src.features.team_chemistry — club-concentration ("chemistry") metrics
for a squad at one tournament: largest/two-club spine, Herfindahl concentration,
same-club teammate pairs, and distinct-club count.

The oracle contrast is the real intuition behind the feature: a squad built on a
single club spine (Germany 2014's Bayern core) vs one scattered across many
clubs — the concentrated squad must score higher on every concentration metric.
Numbers below are worked out by hand from the club-count formulas.
"""

import math

import pytest

from src.features import team_chemistry


def test_concentrated_squad_hand_computed(make_squads):
    """
    4 players: 3 at Bayern, 1 at Dortmund. n=4.
      largest_club_bloc    = 3
      top2_club_bloc       = 3 + 1 = 4
      n_distinct_clubs     = 2
      club_hhi             = (3/4)^2 + (1/4)^2 = 0.5625 + 0.0625 = 0.625
      same_club_pairs      = C(3,2) + C(1,2) = 3 + 0 = 3
      same_club_pair_ratio = 3 / C(4,2) = 3 / 6 = 0.5
    """
    squads = make_squads(
        [
            {"team": "Germany", "club": "Bayern"},
            {"team": "Germany", "club": "Bayern"},
            {"team": "Germany", "club": "Bayern"},
            {"team": "Germany", "club": "Dortmund"},
        ]
    )
    out = team_chemistry(squads, "Germany", "World Cup 2014")

    assert out["largest_club_bloc"] == 3
    assert out["top2_club_bloc"] == 4
    assert out["n_distinct_clubs"] == 2
    assert out["club_hhi"] == pytest.approx(0.625)
    assert out["same_club_pairs"] == 3
    assert out["same_club_pair_ratio"] == pytest.approx(0.5)


def test_scattered_squad_is_minimally_concentrated(make_squads):
    """
    4 players at 4 different clubs. n=4.
      largest = 1, top2 = 2, n_distinct = 4
      hhi = 4 * (1/4)^2 = 0.25   (the 1/n floor)
      same_club_pairs = 0, ratio = 0.0
    """
    squads = make_squads(
        [
            {"team": "Germany", "club": "ClubA"},
            {"team": "Germany", "club": "ClubB"},
            {"team": "Germany", "club": "ClubC"},
            {"team": "Germany", "club": "ClubD"},
        ]
    )
    out = team_chemistry(squads, "Germany", "World Cup 2014")

    assert out["largest_club_bloc"] == 1
    assert out["n_distinct_clubs"] == 4
    assert out["club_hhi"] == pytest.approx(0.25)
    assert out["same_club_pairs"] == 0
    assert out["same_club_pair_ratio"] == pytest.approx(0.0)


def test_concentration_ordering_matches_intuition(make_squads):
    """The concentrated squad must out-score the scattered one on HHI and pair
    ratio — the whole point of the feature."""
    concentrated = make_squads([{"team": "G", "club": "Bayern"} for _ in range(4)])
    scattered = make_squads([{"team": "G", "club": f"C{i}"} for i in range(4)])

    c = team_chemistry(concentrated, "G", "World Cup 2014")
    s = team_chemistry(scattered, "G", "World Cup 2014")

    assert c["club_hhi"] > s["club_hhi"]
    assert c["same_club_pair_ratio"] > s["same_club_pair_ratio"]


def test_unknown_squad_returns_zero_counts_but_nan_ratios(make_squads):
    """No matching squad → counts a true 0, but hhi and pair ratio NaN (undefined,
    never guessed)."""
    squads = make_squads([{"team": "Germany", "club": "Bayern"}])
    out = team_chemistry(squads, "Brazil", "World Cup 2014")

    assert out["largest_club_bloc"] == 0
    assert out["n_distinct_clubs"] == 0
    assert math.isnan(out["club_hhi"])
    assert math.isnan(out["same_club_pair_ratio"])
