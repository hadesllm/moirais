"""Tests for morie.fn.r_es -- Pearson r as an effect size."""

import pytest
from morie.fn.r_es import r_effect_size
from morie.fn._containers import ESRes


class TestREffectSize:
    def test_perfect_positive(self):
        """Perfectly correlated data gives r = 1."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = r_effect_size(x, y)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative(self):
        """Perfectly negatively correlated gives r = -1."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        result = r_effect_size(x, y)
        assert result.estimate == pytest.approx(-1.0, abs=1e-10)

    def test_ci_bounds_ordered(self):
        """CI lower <= estimate <= CI upper."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [1.2, 2.5, 2.9, 4.1, 5.3, 5.8, 7.2, 8.1]
        result = r_effect_size(x, y)
        assert result.ci_lower <= result.estimate <= result.ci_upper
