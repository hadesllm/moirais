"""Tests for morie.fn.omega2 -- Omega-squared from ANOVA."""

import pytest
from morie.fn.omega2 import omega_squared


class TestOmegaSquared:
    def test_known_value(self):
        """omega2 = df_b*(F-1) / (df_b*F + df_w + 1)."""
        # F=10, df_between=2, df_within=97, n=100
        expected = (2 * (10 - 1)) / (2 * 10 + 97 + 1)  # 18/118 ~ 0.1525
        assert omega_squared(10, 2, 97, 100) == pytest.approx(expected, abs=1e-10)

    def test_smaller_than_eta2(self):
        """Omega-squared is always <= eta-squared for the same data."""
        from morie.fn.eta2 import eta_squared
        e2 = eta_squared(10, 2, 97)
        o2 = omega_squared(10, 2, 97, 100)
        assert o2 <= e2

    def test_clipped_to_zero(self):
        """F close to 1 with many groups can give omega2 <= 0, clipped to 0."""
        result = omega_squared(0.5, 3, 96, 100)
        assert result >= 0.0
