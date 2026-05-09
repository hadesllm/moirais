"""Tests for moirais.fn.pwr_t -- power analysis for t-tests."""

import pytest
from moirais.fn.pwr_t import power_t_test


class TestPowerTTest:
    def test_solve_power_medium_effect(self):
        """d=0.5, n=64 per group should give power near 0.80."""
        pwr = power_t_test(n=64, delta=0.5, sd=1.0, power=None)
        assert 0.70 < pwr < 0.90

    def test_solve_n(self):
        """Solve for n given d=0.5, power=0.80."""
        n = power_t_test(n=None, delta=0.5, sd=1.0, power=0.80)
        assert isinstance(n, float)
        assert 50 < n < 80

    def test_exactly_one_none_required(self):
        with pytest.raises(ValueError, match="Exactly one"):
            power_t_test(n=64, delta=0.5, sd=1.0, power=0.8)
