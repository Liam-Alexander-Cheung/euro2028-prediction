"""
Tests for src.odds — the bookmaker-odds maths used as one of the two reference
forecasts the WDL model is judged against.

These promote the module's own `_self_test()` oracles into real pytest cases:
de-vigging (strip the bookmaker margin so probabilities sum to 1),
American→decimal conversion (incl. the bare-integer paste bug), and the
conservative team-name crosswalk.
"""

import math

import pytest

from src.odds import american_to_decimal, canonical_team, devig


class TestDevig:
    def test_fair_odds_are_left_essentially_unchanged(self):
        # 2.0 / 4.0 / 4.0 imply 0.50 / 0.25 / 0.25, already summing to 1 → identity
        p = devig(2.0, 4.0, 4.0)
        assert sum(p) == pytest.approx(1.0, abs=1e-12)
        assert p[0] == pytest.approx(0.50)
        assert p[1] == pytest.approx(0.25)
        assert p[2] == pytest.approx(0.25)

    def test_margin_is_removed_and_favourite_preserved(self):
        # a real vig-carrying market: raw implied sum > 1, output sums to 1, and
        # the home favourite stays the largest probability
        raw_sum = 1 / 1.90 + 1 / 3.50 + 1 / 4.20
        assert raw_sum > 1.0  # there IS a margin to strip
        p = devig(1.90, 3.50, 4.20)
        assert sum(p) == pytest.approx(1.0, abs=1e-12)
        assert p[0] > p[1] and p[0] > p[2]

    def test_heavy_favourite_gets_high_probability(self):
        p = devig(1.04, 15.0, 41.0)
        assert p[0] > 0.90

    @pytest.mark.parametrize("bad", [
        (float("nan"), 3.5, 4.2),
        (1.0, 3.5, 4.2),        # an odd of exactly 1.0 is impossible in a real market
        (None, 3.5, 4.2),
    ])
    def test_invalid_odds_return_all_nan(self, bad):
        result = devig(*bad)
        assert all(math.isnan(x) for x in result)


class TestAmericanToDecimal:
    def test_signed_american(self):
        assert american_to_decimal("+140") == pytest.approx(2.40)
        assert american_to_decimal("-175") == pytest.approx(1.0 + 100 / 175)
        assert american_to_decimal("+100") == pytest.approx(2.00)   # even money

    def test_already_decimal_passes_through(self):
        assert american_to_decimal("1.44") == pytest.approx(1.44)

    def test_bare_unsigned_integer_is_treated_as_positive_american(self):
        # the real paste bug: a stripped "+" must NOT read as a decimal 260.0
        assert american_to_decimal("260") == pytest.approx(3.60)
        assert american_to_decimal("425") == pytest.approx(5.25)

    def test_comma_thousands_separator_tolerated(self):
        assert american_to_decimal("+1,200") == pytest.approx(13.0)

    @pytest.mark.parametrize("bad", ["", None, "nan", "0", "+0"])
    def test_unparseable_returns_nan(self, bad):
        assert math.isnan(american_to_decimal(bad))


class TestCanonicalTeam:
    DB_NAMES = {"Germany", "Republic of Ireland", "Bosnia and Herzegovina"}

    def test_hand_verified_alias_resolves(self):
        # the "Ireland" trap: it maps to the Republic, never Northern Ireland
        assert canonical_team("Ireland", self.DB_NAMES) == "Republic of Ireland"
        assert canonical_team("Bosnia & Herzegovina", self.DB_NAMES) == "Bosnia and Herzegovina"

    def test_exact_db_name_passes_through(self):
        assert canonical_team("Germany", self.DB_NAMES) == "Germany"

    def test_unresolved_name_returns_none_not_a_fuzzy_guess(self):
        # unknown → None (caller queues for human review); never an auto fuzzy match
        assert canonical_team("Someland", self.DB_NAMES) is None
