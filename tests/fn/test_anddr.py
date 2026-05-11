"""Tests for morie.fn.anddr -- Anderson-Darling test."""

import numpy as np
import pytest
from morie.fn.anddr import anderson_darling


class TestAndersonDarling:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        r = anderson_darling(x)
        assert r["A2"] > 0
        assert "15.0" in r["critical_values"]

    def test_non_normal_rejects(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1.0, 200)
        r = anderson_darling(x, dist="norm")
        assert r["significance_level"] is not None

    def test_exponential_dist(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1.0, 200)
        r = anderson_darling(x, dist="expon")
        assert r["A2"] > 0

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 3"):
            anderson_darling(np.array([1.0, 2.0]))

    def test_unknown_dist(self):
        with pytest.raises(ValueError, match="dist must be"):
            anderson_darling(np.array([1, 2, 3, 4, 5]), dist="bad")
