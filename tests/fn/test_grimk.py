"""Tests for morie.fn.grimk -- power law fit."""

import numpy as np
from morie.fn.grimk import power_law_fit, grimk
from morie.fn._containers import DescriptiveResult


class TestGrimk:
    def test_alias(self):
        assert grimk is power_law_fit

    def test_exact_power_law(self):
        x = np.arange(1, 50, dtype=float)
        y = 2.5 * x ** 1.5
        r = power_law_fit(x, y)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value["b"] - 1.5) < 0.1
        assert r.extra["r_squared"] > 0.99

    def test_negative_values_filtered(self):
        x = np.array([-1, 0, 1, 2, 3, 4, 5], dtype=float)
        y = np.array([-1, 0, 1, 4, 9, 16, 25], dtype=float)
        r = power_law_fit(x, y)
        assert r.extra["n"] == 5
