"""Tests for morie.fn.mouse -- Monte Carlo integration."""

import numpy as np
from morie.fn.mouse import monte_carlo, mouse
from morie.fn._containers import DescriptiveResult


class TestMouse:
    def test_alias(self):
        assert mouse is monte_carlo

    def test_x_squared(self):
        result = monte_carlo("x**2", lo=0, hi=1, n_samples=50000, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 1 / 3) < 0.02

    def test_sin(self):
        result = monte_carlo("np.sin(x)", lo=0, hi=np.pi, n_samples=50000, seed=42)
        assert abs(result.value - 2.0) < 0.05
