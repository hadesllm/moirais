"""Tests for moirais.fn.r2 -- R-squared (coefficient of determination)."""

import numpy as np
import pytest
from moirais.fn.r2 import r_squared
from moirais.fn._containers import ESRes


class TestRSquared:
    def test_perfect_correlation(self):
        """Perfectly correlated data gives R2 = 1."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = r_squared(x, y)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(1.0, abs=1e-10)

    def test_uncorrelated_near_zero(self, rng):
        """Uncorrelated data gives R2 near 0."""
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        result = r_squared(x, y)
        assert result.estimate < 0.1

    def test_bounded_zero_one(self):
        """R2 should be in [0, 1]."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.5, 2.5, 2.8, 4.2]
        result = r_squared(x, y)
        assert 0.0 <= result.estimate <= 1.0
