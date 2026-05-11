"""Tests for morie.fn.eps2 -- Epsilon-squared (Kelley, 1935)."""

import pytest
from morie.fn.eps2 import epsilon_squared
from morie.fn._containers import ESRes


class TestEpsilonSquared:
    def test_known_value(self):
        """eps2 = (SS_effect - df_effect * MS_error) / SS_total."""
        result = epsilon_squared(ss_effect=30.0, ss_total=100.0, df_effect=2, ms_error=5.0)
        assert isinstance(result, ESRes)
        expected = (30.0 - 2 * 5.0) / 100.0  # 20/100 = 0.2
        assert result.estimate == pytest.approx(expected, abs=1e-10)

    def test_clipped_to_zero(self):
        """When effect is small relative to error, clip to 0."""
        result = epsilon_squared(ss_effect=2.0, ss_total=100.0, df_effect=5, ms_error=10.0)
        assert result.estimate >= 0.0

    def test_zero_total_returns_zero(self):
        """Zero SS total should return 0."""
        result = epsilon_squared(ss_effect=0.0, ss_total=0.0, df_effect=1, ms_error=0.0)
        assert result.estimate == pytest.approx(0.0)
