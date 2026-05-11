"""Tests for morie.fn.spln -- Cubic spline regression."""

import numpy as np
from morie.fn.spln import spline_regression, spln
from morie.fn._containers import DescriptiveResult


class TestSpln:
    def test_alias(self):
        assert spln is spline_regression

    def test_smooth_data(self):
        x = np.linspace(0, 2 * np.pi, 50)
        y = np.sin(x)
        result = spline_regression(x, y, n_knots=8)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["r_squared"] > 0.9

    def test_returns_predicted(self):
        x = np.linspace(0, 1, 30)
        y = x ** 2
        result = spline_regression(x, y)
        assert len(result.extra["predicted"]) == 30
