"""
Tests for src.features.squad_age_depth — the age + positional-depth profile of a
team's squad at ONE tournament (squad-scoped, not date-scoped). Returns mean age,
squad size, per-position counts, and per-position proportions, always keyed by
the four known positions GK/DF/MF/FW.
"""

import math

import pytest

from src.features import squad_age_depth


def test_counts_proportions_and_mean_age(make_squads):
    """
    A 4-player squad: GK(25), DF(26), DF(27), MF(30).
      mean_age = (25+26+27+30)/4 = 27.0
      counts   = GK 1, DF 2, MF 1, FW 0
      props    = GK .25, DF .5, MF .25, FW 0.0   (count / squad_size)
    """
    squads = make_squads(
        [
            {"team": "Germany", "position": "GK", "age_at_tournament": 25},
            {"team": "Germany", "position": "DF", "age_at_tournament": 26},
            {"team": "Germany", "position": "DF", "age_at_tournament": 27},
            {"team": "Germany", "position": "MF", "age_at_tournament": 30},
        ]
    )
    out = squad_age_depth(squads, "Germany", "World Cup 2014")

    assert out["squad_size"] == 4
    assert out["mean_age"] == pytest.approx(27.0)
    assert out["position_counts"] == {"GK": 1, "DF": 2, "MF": 1, "FW": 0}
    assert out["position_proportions"] == {"GK": 0.25, "DF": 0.5, "MF": 0.25, "FW": 0.0}


def test_missing_positions_are_present_as_zero(make_squads):
    """A squad with no forwards still reports FW: 0 (fixed 4-key shape), not a
    missing key — so every squad's dict is directly comparable."""
    squads = make_squads(
        [
            {"team": "Germany", "position": "GK"},
            {"team": "Germany", "position": "DF"},
        ]
    )
    out = squad_age_depth(squads, "Germany", "World Cup 2014")
    assert set(out["position_counts"]) == {"GK", "DF", "MF", "FW"}
    assert out["position_counts"]["FW"] == 0
    assert out["position_counts"]["MF"] == 0


def test_unknown_squad_returns_zero_counts_but_nan_proportions(make_squads):
    """
    A team/tournament that matches no squad: counts are a true 0 (a real fact
    about an empty result), but proportions are NaN — "0 out of 0" is undefined,
    never a guessed 0.0.
    """
    squads = make_squads([{"team": "Germany", "position": "GK"}])
    out = squad_age_depth(squads, "Brazil", "World Cup 2014")

    assert out["squad_size"] == 0
    assert math.isnan(out["mean_age"])
    assert out["position_counts"] == {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
    assert all(math.isnan(v) for v in out["position_proportions"].values())
