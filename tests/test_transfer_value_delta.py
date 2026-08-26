"""
Tests for src.features.transfer_value_delta_z — returns a player's raw market
value at two points in time (now, and `months_back` earlier), each picked as the
most recent valuation ON OR BEFORE the target date.

The real function calls Transfermarkt over the network (and is currently WAF-
blocked anyway). We don't want the network in a unit test, so we monkeypatch the
`fetch_market_value_history` it pulls in and feed it a known valuation history —
then assert the date-selection logic on the numbers we control.
"""

import math

import pandas as pd

from src.features import transfer_value_delta_z

# A fixed valuation history in Transfermarkt's own shape: `datum_mw` is a
# day/month/year string, `y` is the value in euros.
_HISTORY = [
    {"datum_mw": "01/01/2019", "y": 10_000_000},
    {"datum_mw": "01/01/2020", "y": 20_000_000},
    {"datum_mw": "01/06/2020", "y": 30_000_000},
]


def test_picks_closest_valuation_at_or_before_each_target(monkeypatch):
    """
    as_of = 2020-07-01, months_back = 12 (target_past ≈ 2019-07-07):
      recent → most recent point ≤ 2020-07-01  = 01/06/2020 → 30M
      past   → most recent point ≤ 2019-07-07  = 01/01/2019 → 10M
    (the 01/01/2020 point is AFTER the past target, so it must not be chosen).
    """
    # patch where the function looks it up (it imports from src.data_pipeline at call time)
    monkeypatch.setattr("src.data_pipeline.fetch_market_value_history", lambda pid: _HISTORY)

    recent, past = transfer_value_delta_z("anyid", pd.Timestamp("2020-07-01"), months_back=12)
    assert recent == 30_000_000
    assert past == 10_000_000


def test_no_valuation_before_target_returns_nan(monkeypatch):
    """A target date earlier than every recorded valuation → NaN, never a
    back-fabricated value."""
    monkeypatch.setattr("src.data_pipeline.fetch_market_value_history", lambda pid: _HISTORY)

    recent, past = transfer_value_delta_z("anyid", pd.Timestamp("2018-01-01"), months_back=12)
    assert math.isnan(recent)
    assert math.isnan(past)
