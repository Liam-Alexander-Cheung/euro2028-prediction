"""
Shared pytest fixtures and hooks for the StatXI test suite.

`conftest.py` is a special filename pytest looks for automatically: anything
defined here is available to every test file in this directory without an
import. We use it for two things:

  1. `make_matches` — a factory fixture that turns a list of plain dicts into a
     matches DataFrame shaped exactly like the one src/features.py expects
     (the columns real match data carries: date, home/away team, scores,
     tournament). Tests hand it a handful of rows with KNOWN outcomes so the
     expected feature value can be worked out by hand — no 30 MB database, no
     network, fully deterministic.

  2. A collection hook that auto-skips any test tagged `@pytest.mark.needs_db`
     when data/statxi.db is absent, so an opt-in "check against the real data"
     test can live beside the synthetic ones without making the whole suite
     depend on a file that is deliberately never committed.
"""

import pandas as pd
import pytest

# DB_PATH is the single source of truth for where the (regenerable) SQLite
# database lives; reusing it means the skip logic can never drift from the
# real path the app uses.
from src.database import DB_PATH


@pytest.fixture
def make_matches():
    """
    Return a factory: call it with a list of row-dicts to get a matches
    DataFrame with the right columns and a real datetime `date` column.

    Each row dict may specify `date`, `home_team`, `away_team`, `home_score`,
    `away_score`, `tournament`. Missing keys fall back to sensible defaults
    (tournament="Friendly", scores=0) so a test only has to state the fields
    it actually cares about.

    Example (a 2-1 home win in a friendly):
        matches = make_matches([
            {"date": "2020-01-01", "home_team": "A", "away_team": "B",
             "home_score": 2, "away_score": 1},
        ])
    """
    columns = ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]

    def _make(rows: list[dict]) -> pd.DataFrame:
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_score": row.get("home_score", 0),
                    "away_score": row.get("away_score", 0),
                    # default to a real tournament label importance_weight knows,
                    # so weights are well-defined without every test spelling it out
                    "tournament": row.get("tournament", "Friendly"),
                }
            )
        df = pd.DataFrame(normalized, columns=columns)
        # features.py compares `date` against pd.Timestamp values, so the column
        # must be real datetimes, not the strings the tests pass in for brevity
        df["date"] = pd.to_datetime(df["date"])
        return df

    return _make


@pytest.fixture
def make_squads():
    """
    Return a factory for a squads DataFrame shaped like the one the squad-scoped
    features (squad_age_depth, team_chemistry) consume: one row per player with
    `team, tournament_name, position, age_at_tournament, club`.

    Call it with a list of row-dicts; only the fields a test cares about need
    spelling out (position defaults to "MF", age to 27, club to a unique-ish
    placeholder, tournament_name to "World Cup 2014").

    Example (two Bayern players in Germany's 2014 squad):
        squads = make_squads([
            {"team": "Germany", "position": "DF", "club": "Bayern Munich"},
            {"team": "Germany", "position": "MF", "club": "Bayern Munich"},
        ])
    """
    columns = ["team", "tournament_name", "position", "age_at_tournament", "club"]

    def _make(rows: list[dict]) -> pd.DataFrame:
        normalized = []
        for i, row in enumerate(rows):
            normalized.append(
                {
                    "team": row["team"],
                    "tournament_name": row.get("tournament_name", "World Cup 2014"),
                    "position": row.get("position", "MF"),
                    "age_at_tournament": row.get("age_at_tournament", 27),
                    # a per-row default club keeps every player at a distinct club
                    # unless a test deliberately shares one (the chemistry signal)
                    "club": row.get("club", f"Club{i}"),
                }
            )
        return pd.DataFrame(normalized, columns=columns)

    return _make


@pytest.fixture(scope="session")
def real_matches():
    """
    The cleaned real match table, loaded once per test session.

    Session-scoped so the ~14s load/clean happens a single time even though
    several `needs_db` tests use it. Because a fixture only runs when a test
    actually requests it, a DB-free run (where every `needs_db` test is skipped)
    never triggers this at all.
    """
    from src.data_pipeline import clean_matches, load_raw_matches

    return clean_matches(load_raw_matches())


def pytest_collection_modifyitems(config, items):
    """
    pytest calls this once after collecting all tests. When the regenerable
    database is missing, we attach a skip marker to every test tagged
    `needs_db`, so those tests are cleanly reported as skipped rather than
    erroring on a missing file. When the DB IS present (as it is on the
    author's machine), they run for real.
    """
    if DB_PATH.exists():
        return
    skip_no_db = pytest.mark.skip(
        reason="needs data/statxi.db (regenerable via `make migrate`)"
    )
    for item in items:
        if "needs_db" in item.keywords:
            item.add_marker(skip_no_db)
