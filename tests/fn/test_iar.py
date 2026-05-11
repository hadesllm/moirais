"""Tests for morie.fn.iar -- indirect age-adjustment (SMR)."""

import numpy as np
import pytest
from morie.fn.iar import indirect_age_adjustment


class TestIAR:
    def test_smr(self):
        """Simple SMR: O=50, reference rates applied to pop => E."""
        obs = np.array([20, 30])
        ref_rates = np.array([0.005, 0.010])
        pop = np.array([3000, 2000])
        res = indirect_age_adjustment(obs, ref_rates, pop)
        assert res.measure == "SMR"
        E = 0.005 * 3000 + 0.010 * 2000
        expected_smr = 50 / E
        assert res.estimate == pytest.approx(expected_smr, rel=0.01)

    def test_ci_bounds(self):
        """CI should bracket the estimate."""
        obs = np.array([20, 30])
        ref_rates = np.array([0.005, 0.010])
        pop = np.array([3000, 2000])
        res = indirect_age_adjustment(obs, ref_rates, pop)
        assert res.ci_lower < res.estimate
        assert res.ci_upper > res.estimate

    def test_extra_fields(self):
        """extra should contain O and E."""
        obs = np.array([10, 15])
        ref_rates = np.array([0.01, 0.02])
        pop = np.array([1000, 500])
        res = indirect_age_adjustment(obs, ref_rates, pop)
        assert res.extra["O"] == pytest.approx(25.0)
        assert res.extra["E"] == pytest.approx(20.0)
