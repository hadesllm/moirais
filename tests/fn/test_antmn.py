"""Tests for moirais.fn.antmn -- allometric regression."""

import numpy as np
from moirais.fn.antmn import allometric_regression, antmn
from moirais.fn._containers import RegressionResult


class TestAntmn:
    def test_alias(self):
        assert antmn is allometric_regression

    def test_known_scaling(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 100, 50)
        y = 2.5 * x ** 0.75 * rng.lognormal(0, 0.05, 50)
        r = allometric_regression(x, y)
        assert isinstance(r, RegressionResult)
        assert abs(r.coefficients["b"] - 0.75) < 0.1
        assert r.r_squared > 0.9

    def test_extra_fields(self):
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        y = x ** 2
        r = allometric_regression(x, y)
        assert "scaling_law" in r.extra
