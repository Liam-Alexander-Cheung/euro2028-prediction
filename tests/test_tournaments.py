"""
Tests for src.tournaments — loading and validating the hand-encoded tournament
structure the Monte Carlo simulator needs (which teams in which group, how the
knockout bracket is wired). The four real configs are committed source data, so
they double as fixtures here.

We check: the real configs load and self-validate; team enumeration is correct;
`validate_teams` fails loudly on an unknown team (the never-fabricate rule); and
the structural validators reject malformed configs.
"""

import pytest

from src.tournaments import (
    all_config_teams,
    load_tournament_config,
    validate_teams,
)
from src.tournaments import _validate_structure  # internal, but the core guard worth pinning


class TestLoadRealConfigs:
    @pytest.mark.parametrize("name,fmt,n_teams", [
        ("wc2022", "wc32", 32),
        ("euro2016", "euro24", 24),
        ("euro2020", "euro24", 24),
        ("euro2024", "euro24", 24),
    ])
    def test_committed_config_loads_and_self_validates(self, name, fmt, n_teams):
        # load_tournament_config runs _validate_structure internally, so a clean
        # load already proves the config's internal arithmetic is consistent
        cfg = load_tournament_config(name)
        assert cfg["format"] == fmt
        teams = all_config_teams(cfg)
        assert len(teams) == n_teams
        assert len(set(teams)) == n_teams  # all distinct — no duplicate team name

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_tournament_config("no_such_tournament")


class TestValidateTeams:
    def _strengths_for(self, cfg, drop=None):
        # validate_teams only inspects strengths["attack"].keys()
        teams = [t for t in all_config_teams(cfg) if t != drop]
        return {"attack": {t: 0.0 for t in teams}}

    def test_passes_when_every_team_has_a_strength(self):
        cfg = load_tournament_config("euro2024")
        # should not raise
        validate_teams(cfg, self._strengths_for(cfg))

    def test_raises_when_a_team_is_missing_from_the_fit(self):
        cfg = load_tournament_config("euro2024")
        missing = all_config_teams(cfg)[0]
        with pytest.raises(ValueError) as exc:
            validate_teams(cfg, self._strengths_for(cfg, drop=missing))
        assert missing in str(exc.value)  # the offending name is surfaced, not swallowed


class TestStructuralValidators:
    def test_unsupported_format_is_rejected(self):
        with pytest.raises(ValueError):
            _validate_structure({"format": "league38"})

    def test_wrong_group_size_is_rejected(self):
        # a euro24 group with 3 teams instead of 4 must fail loudly
        bad = {
            "format": "euro24",
            "groups": {g: ["t1", "t2", "t3"] for g in "ABCDEF"},
        }
        with pytest.raises(ValueError):
            _validate_structure(bad)
