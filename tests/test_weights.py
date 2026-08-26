"""
Tests for the match-weighting helpers in src.data_pipeline that every match
feature depends on:

  * importance_weight   — tournament tier → multiplier (explicit allowlists)
  * recency_weight      — exponential time decay, floored at min_weight
  * fifa_edition_for_date — era-correct FIFA/FC edition for a date (leakage guard)

These are the shared foundation under rolling_form / head_to_head / goal_trend,
so a silent change here would move every feature at once — exactly what a test
should pin down.
"""

import pandas as pd
import pytest

from src.data_pipeline import (
    fifa_edition_for_date,
    importance_weight,
    recency_weight,
)


class TestImportanceWeight:
    def test_tier_values(self):
        # one representative name per tier, checked against the allowlists
        assert importance_weight("FIFA World Cup") == 1.0          # major final
        assert importance_weight("UEFA Euro") == 1.0
        assert importance_weight("FIFA World Cup qualification") == 0.6
        assert importance_weight("UEFA Nations League") == 0.55    # 2nd-tier confed
        assert importance_weight("AFF Championship") == 0.45       # regional
        assert importance_weight("Asian Games") == 0.2             # multisport
        assert importance_weight("Friendly") == 0.3

    def test_unknown_tournament_falls_to_long_tail_default(self):
        # obscure invitationals aren't in any allowlist → 0.25, not a crash
        assert importance_weight("Some Obscure Invitational Cup") == 0.25

    def test_tier_ordering(self):
        # the tiers must stay strictly ranked: a World Cup outweighs its own
        # qualifiers, which outweigh a friendly
        assert (
            importance_weight("FIFA World Cup")
            > importance_weight("FIFA World Cup qualification")
            > importance_weight("Friendly")
        )


class TestRecencyWeight:
    REF = pd.Timestamp("2020-01-01")

    def test_same_day_is_full_weight(self):
        assert recency_weight(self.REF, self.REF, half_life_days=730) == pytest.approx(1.0)

    def test_one_half_life_halves_the_weight(self):
        one_hl = self.REF - pd.Timedelta(days=730)
        assert recency_weight(one_hl, self.REF, half_life_days=730) == pytest.approx(0.5)

    def test_two_half_lives_quarter_the_weight(self):
        two_hl = self.REF - pd.Timedelta(days=1460)
        assert recency_weight(two_hl, self.REF, half_life_days=730) == pytest.approx(0.25)

    def test_floor_prevents_ancient_matches_vanishing(self):
        # 20 years back at a 2-year half-life would decay to ~1e-3; the 0.05 floor
        # catches it, honouring the "1990-onward data is worth keeping" decision
        ancient = self.REF - pd.Timedelta(days=7300)
        assert recency_weight(ancient, self.REF, half_life_days=730) == pytest.approx(0.05)

    def test_custom_floor_is_respected(self):
        ancient = self.REF - pd.Timedelta(days=7300)
        assert recency_weight(ancient, self.REF, half_life_days=730, min_weight=0.1) == pytest.approx(0.1)


class TestFifaEditionForDate:
    def test_none_before_fifa_15(self):
        # the ratings window opens at FIFA 15 (Sep 2014); anything earlier is None
        assert fifa_edition_for_date(pd.Timestamp("2014-08-31")) is None
        assert fifa_edition_for_date(pd.Timestamp("2010-01-01")) is None

    def test_september_release_boundary(self):
        # editions ship in September: Aug 2014 is still pre-window; Sep 2014 = FIFA 15
        assert fifa_edition_for_date(pd.Timestamp("2014-09-01")) == 15
        # before September a year's new edition isn't out yet: mid-2015 is still 15
        assert fifa_edition_for_date(pd.Timestamp("2015-06-01")) == 15
        # from September the new edition is out: late-2015 is 16
        assert fifa_edition_for_date(pd.Timestamp("2015-09-15")) == 16

    def test_clamps_to_24_for_recent_dates(self):
        # FC 24 is the latest edition held; a 2030 match falls back to it, stale
        # but never leaked from a game that didn't exist
        assert fifa_edition_for_date(pd.Timestamp("2023-09-15")) == 24
        assert fifa_edition_for_date(pd.Timestamp("2030-01-01")) == 24

    def test_matches_a_known_in_scope_tournament(self):
        # World Cup 2022 (Nov 2022) should resolve to FIFA 23 — the value the
        # TOURNAMENT_TO_FIFA_VERSION map hard-codes, a nice consistency check
        assert fifa_edition_for_date(pd.Timestamp("2022-11-20")) == 23
