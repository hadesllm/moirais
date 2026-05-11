"""Tests for morie.fn.pwr_av -- power for one-way ANOVA."""

import pytest
from morie.fn.pwr_av import power_anova


class TestPowerAnova:
    def test_solve_power(self):
        """k=3, n=30 per group, f=0.25 should give moderate power."""
        pwr = power_anova(n=30, k=3, f=0.25, power=None)
        assert isinstance(pwr, float)
        assert 0.4 < pwr < 0.95

    def test_solve_n(self):
        """Solve for n with k=3, f=0.25, power=0.80."""
        n = power_anova(n=None, k=3, f=0.25, power=0.80)
        assert isinstance(n, float)
        assert n > 10

    def test_k_less_than_2_raises(self):
        with pytest.raises(ValueError, match="k must be >= 2"):
            power_anova(n=30, k=1, f=0.25, power=None)
