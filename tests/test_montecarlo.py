"""
Tests for src.models.montecarlo — the tournament simulator's sampling core and
end-to-end run.

The headline test promotes the module's `__main__` "Phase A" self-check into a
proper assertion: the Monte Carlo grid-sampler must reproduce the analytic
Dixon-Coles W/D/L probabilities to within Monte Carlo error. Everything here is
seeded, so it's fully deterministic and needs no DB (a synthetic strengths dict
drives the end-to-end simulation).
"""

import numpy as np
import pytest

from src.models.montecarlo import (
    _wdl_from_samples,
    sample_scorelines,
    simulate_tournament,
)
from src.models.poisson import apply_dixon_coles, scoreline_matrix, wdl_from_grid
from src.tournaments import all_config_teams, load_tournament_config


def _uniform_strengths(cfg):
    """Every team identical: base gives ~1 goal each, no home edge, no DC — enough
    for a valid, fast end-to-end simulation without fitting anything."""
    teams = all_config_teams(cfg)
    return {
        "base": 0.0,
        "home_adv": 0.0,
        "attack": {t: 0.0 for t in teams},
        "defence": {t: 0.0 for t in teams},
        "rho": 0.0,
    }


class TestSamplerCore:
    def test_wdl_from_samples_counts_fractions(self):
        # home 2-0 (win), 1-1 (draw), 0-2 (loss) → 1/3 each
        home = np.array([2, 1, 0])
        away = np.array([0, 1, 2])
        assert _wdl_from_samples(home, away) == pytest.approx((1 / 3, 1 / 3, 1 / 3))

    def test_sampler_is_deterministic_under_a_fixed_seed(self):
        grid = scoreline_matrix(1.6, 1.1)
        h1, a1 = sample_scorelines(grid, 1000, np.random.default_rng(0))
        h2, a2 = sample_scorelines(grid, 1000, np.random.default_rng(0))
        assert np.array_equal(h1, h2)
        assert np.array_equal(a1, a2)


class TestPhaseASelfCheck:
    def test_monte_carlo_matches_analytic_within_error(self):
        """The ported Phase-A check: sampling the Dixon-Coles-corrected grid must
        reproduce wdl_from_grid to within ~Monte-Carlo standard error."""
        lam_h, lam_a, rho = 1.6, 1.1, -0.047
        grid = apply_dixon_coles(scoreline_matrix(lam_h, lam_a), lam_h, lam_a, rho)
        analytic = wdl_from_grid(grid)                       # the truth

        n = 500_000
        hg, ag = sample_scorelines(grid, n, np.random.default_rng(0))
        mc = _wdl_from_samples(hg, ag)                       # the estimate

        # standard error of a proportion is sqrt(p(1-p)/n); 5 std errs is a safe
        # non-flaky band (a genuine bug moves the estimate far more than that)
        se = max(np.sqrt(p * (1 - p) / n) for p in analytic)
        for m, a in zip(mc, analytic):
            assert abs(m - a) < 5 * se

    def test_grid_sampler_agrees_with_independent_poisson_when_rho_zero(self):
        """With rho=0 the grid IS the independent product, so the grid-sampler and
        two naive independent Poisson draws must agree on the draw rate."""
        lam_h, lam_a = 1.6, 1.1
        grid0 = scoreline_matrix(lam_h, lam_a)              # rho=0
        rng = np.random.default_rng(0)

        _, gd, _ = _wdl_from_samples(*sample_scorelines(grid0, 500_000, rng))
        _, id_, _ = _wdl_from_samples(rng.poisson(lam_h, 500_000), rng.poisson(lam_a, 500_000))
        assert abs(gd - id_) < 5e-3


class TestSimulateTournament:
    def test_end_to_end_reproducible_and_internally_consistent(self):
        cfg = load_tournament_config("euro2024")
        strengths = _uniform_strengths(cfg)

        df1 = simulate_tournament(cfg, strengths, n=3000, seed=0)
        df2 = simulate_tournament(cfg, strengths, n=3000, seed=0)

        # same seed → identical result (seeded reproducibility)
        from pandas.testing import assert_frame_equal
        assert_frame_equal(df1, df2)

        # exactly one champion per sim → win probabilities sum to 1
        assert df1["p_win"].sum() == pytest.approx(1.0)
        # 16 teams reach the R16 in every sim → reach-probs sum to 16
        assert df1["p_R16"].sum() == pytest.approx(16.0)
        # reaching a later round implies reaching the earlier one (per team)
        assert (df1["p_win"] <= df1["p_final"] + 1e-12).all()
        assert (df1["p_final"] <= df1["p_SF"] + 1e-12).all()
