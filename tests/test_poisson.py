"""
Tests for the Poisson / Dixon-Coles scoreline maths in src.models.poisson.

These target the pure, deterministic pieces (no fitting, no DB): the scoreline
grid, the W/D/L collapse, the Dixon-Coles low-score correction, and predict_match
on a hand-built strengths dict. A `needs_db` test fits on the real data and
promotes the Germany-vs-San-Marino sanity check to an assertion.
"""

import math

import numpy as np
import pytest

from src.models.poisson import (
    apply_dixon_coles,
    dixon_coles_tau,
    predict_match,
    scoreline_matrix,
    wdl_from_grid,
)


class TestScorelineMatrix:
    def test_shape_and_total_mass(self):
        grid = scoreline_matrix(1.6, 1.1)
        assert grid.shape == (11, 11)          # (max_goals+1) square
        # truncated at 10-10, so it's ~1 but not exactly; well within 1e-4
        assert grid.sum() == pytest.approx(1.0, abs=1e-4)

    def test_zero_zero_cell_is_analytic(self):
        # P(0,0) under independence = e^-lh * e^-la = e^-(lh+la)
        lh, la = 1.6, 1.1
        grid = scoreline_matrix(lh, la)
        assert grid[0, 0] == pytest.approx(math.exp(-(lh + la)))


class TestWdlFromGrid:
    def test_hand_computed_2x2_grid(self):
        # grid[i,j]: i=home goals, j=away goals.
        #   home win (i>j): cell [1,0]=0.3
        #   draw    (i==j): [0,0]+[1,1]=0.1+0.4=0.5
        #   away win (i<j): cell [0,1]=0.2
        grid = np.array([[0.1, 0.2], [0.3, 0.4]])
        p_home, p_draw, p_away = wdl_from_grid(grid)
        assert (p_home, p_draw, p_away) == pytest.approx((0.3, 0.5, 0.2))
        assert p_home + p_draw + p_away == pytest.approx(1.0)

    def test_equal_rates_are_symmetric(self):
        # equal lambdas → symmetric grid → P(home win) == P(away win) exactly
        p_home, _, p_away = wdl_from_grid(scoreline_matrix(1.5, 1.5))
        assert p_home == pytest.approx(p_away)


class TestDixonColes:
    def test_rho_zero_is_identity(self):
        grid = scoreline_matrix(1.6, 1.1)
        assert np.array_equal(apply_dixon_coles(grid, 1.6, 1.1, 0.0), grid)

    def test_negative_rho_lifts_draws_and_lowers_split_low_scores(self):
        lh, la, rho = 1.6, 1.1, -0.1
        grid = scoreline_matrix(lh, la)
        g = apply_dixon_coles(grid, lh, la, rho)
        # rho<0 raises 0-0 and 1-1 (more draws), lowers 1-0 and 0-1
        assert g[0, 0] > grid[0, 0]
        assert g[1, 1] > grid[1, 1]
        assert g[0, 1] < grid[0, 1]
        assert g[1, 0] < grid[1, 0]
        # every other cell is untouched
        assert g[2, 2] == grid[2, 2]

    def test_tau_factors(self):
        # vectorised tau on the diagonal (0,0),(1,1),(2,2) with matching lambdas
        lh = np.full(3, 1.6)
        la = np.full(3, 1.1)
        tau = dixon_coles_tau(np.array([0, 1, 2]), np.array([0, 1, 2]), lh, la, -0.05)
        assert tau[0] == pytest.approx(1.0 - 1.6 * 1.1 * (-0.05))  # (0,0)
        assert tau[1] == pytest.approx(1.0 - (-0.05))             # (1,1)
        assert tau[2] == pytest.approx(1.0)                       # outside the 4 cells


class TestPredictMatch:
    # a hand-built model: "Strong" out-attacks and out-defends "Weak"
    STRENGTHS = {
        "base": 0.0,
        "home_adv": 0.3,
        "attack": {"Strong": 1.0, "Weak": -1.0, "Even": 0.0},
        "defence": {"Strong": 1.0, "Weak": -1.0, "Even": 0.0},
        "rho": 0.0,
    }

    def test_heavy_favourite_dominates(self):
        # Strong vs Weak at a neutral venue: lambda_home=e^2, lambda_away=e^-2 →
        # a near-certain home win. This is the synthetic Germany-vs-San-Marino case.
        pred = predict_match(self.STRENGTHS, "Strong", "Weak", neutral=True)
        assert pred["lambda_home"] == pytest.approx(math.exp(2.0))
        assert pred["lambda_away"] == pytest.approx(math.exp(-2.0))
        assert pred["p_home"] > 0.95
        assert pred["p_home"] > pred["p_away"]
        assert pred["p_home"] + pred["p_draw"] + pred["p_away"] == pytest.approx(1.0)
        assert pred["top_scoreline"][0] > pred["top_scoreline"][1]

    def test_home_advantage_breaks_a_mirror_matchup(self):
        # two identical teams: neutral → symmetric; at home → home favoured
        neutral = predict_match(self.STRENGTHS, "Even", "Even", neutral=True)
        home = predict_match(self.STRENGTHS, "Even", "Even", neutral=False)
        assert neutral["p_home"] == pytest.approx(neutral["p_away"])
        assert home["p_home"] > home["p_away"]

    def test_unknown_team_returns_none(self):
        assert predict_match(self.STRENGTHS, "Strong", "Nowhereland") is None


@pytest.mark.needs_db
def test_real_fit_germany_beats_san_marino(real_matches):
    """Fit Dixon-Coles on the real match database and assert the archetypal
    mismatch: Germany crushes San Marino. Auto-skips without data/statxi.db."""
    from src.models.poisson import fit_dixon_coles

    strengths = fit_dixon_coles(real_matches)

    pred = predict_match(strengths, "Germany", "San Marino", neutral=True)
    assert pred is not None
    assert pred["lambda_home"] > pred["lambda_away"]
    assert pred["p_home"] > 0.9
