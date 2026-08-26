"""
Tests for clean_player_name in build_squad_schema.py — the squad scraper's
name/captain cleaner.

Regression guard for a real, verified bug: Wikipedia marks the captain two ways
("(captain)" spelled out, and the "(c)" abbreviation on older pages), but an
earlier version stripped only "(captain)". That left 128 "(c)" captains in the
DB unflagged with the marker still in their name (e.g. "Marcel Desailly (c)").
This pins both markers — and pins that an unrelated parenthetical like "(coach)"
is NOT mistaken for a captaincy.

Pure string function, no DB (the scraper's DB work lives in main(), guarded by
__main__), so importing it here is side-effect-free.
"""

from build_squad_schema import clean_player_name


class TestCleanPlayerName:
    def test_spelled_out_captain_marker(self):
        name, is_captain = clean_player_name("Bastian Schweinsteiger (captain)")
        assert name == "Bastian Schweinsteiger"
        assert is_captain is True

    def test_abbreviated_captain_marker(self):
        # the "(c)" form older Wikipedia pages use — the case previously missed
        name, is_captain = clean_player_name("Marcel Desailly (c)")
        assert name == "Marcel Desailly"
        assert is_captain is True

    def test_uppercase_marker_is_handled(self):
        # matching is case-insensitive, so "(C)" is caught too
        name, is_captain = clean_player_name("Some Captain (C)")
        assert name == "Some Captain"
        assert is_captain is True

    def test_non_captain_is_unflagged_and_unchanged(self):
        name, is_captain = clean_player_name("Lionel Messi")
        assert name == "Lionel Messi"
        assert is_captain is False

    def test_coach_parenthetical_is_not_a_captain(self):
        # "(coach)" must not trip the "(c)" rule — the token must be exactly "(c)"
        _, is_captain = clean_player_name("Someone (coach)")
        assert is_captain is False

    def test_trailing_footnote_markers_still_stripped(self):
        # existing behaviour preserved: trailing * / dagger footnotes removed
        name, _ = clean_player_name("Some Player*")
        assert name == "Some Player"

    def test_non_string_input_is_guarded(self):
        # None / NaN player names must not crash the scraper
        assert clean_player_name(None) == (None, False)
