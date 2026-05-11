"""Tests for morie.fn.ad -- Anderson-Darling test for normality."""

import numpy as np
from morie.fn.ad import anderson_darling, ad
from morie.fn._containers import TestResult


class TestAD:
    def test_alias(self):
        assert ad is anderson_darling

    def test_normal_data(self):
        """Normal data should not be rejected."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        result = anderson_darling(x)
        assert isinstance(result, TestResult)
        assert result.test_name == "Anderson-Darling"
        assert result.p_value >= 0.10  # approximate p, should not reject

    def test_uniform_data_rejected(self):
        """Uniform data should be rejected for normality."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-3, 3, 200)
        result = anderson_darling(x)
        assert result.p_value <= 0.05

    def test_extra_has_critical_values(self):
        rng = np.random.default_rng(42)
        result = anderson_darling(rng.normal(0, 1, 100))
        assert "critical_values" in result.extra
        assert "significance_levels" in result.extra
        assert len(result.extra["critical_values"]) > 0

    def test_handles_nan(self):
        x = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        result = anderson_darling(x)
        assert result.n == 9
