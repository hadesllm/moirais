"""Tests for morie.fn.pwr_p -- power for two-proportion z-test."""

import pytest
from morie.fn.pwr_p import power_prop_test


class TestPowerPropTest:
    def test_solve_power(self):
        """p1=0.5, p2=0.3, n=100 should give reasonable power."""
        pwr = power_prop_test(n=100, p1=0.5, p2=0.3, power=None)
        assert isinstance(pwr, float)
        assert 0.5 < pwr < 1.0

    def test_solve_n(self):
        """Solve for n with p1=0.5, p2=0.3, power=0.80."""
        n = power_prop_test(n=None, p1=0.5, p2=0.3, power=0.80)
        assert isinstance(n, float)
        assert n > 30

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            power_prop_test(n=100, p1=1.5, p2=0.3, power=None)
