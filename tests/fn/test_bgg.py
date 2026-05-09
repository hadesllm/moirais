"""Tests for moirais.fn.bgg -- Bagging."""

import numpy as np
from moirais.fn.bgg import bagging, bgg
from moirais.fn._containers import DescriptiveResult


class TestBgg:
    def test_alias(self):
        assert bgg is bagging

    def test_linear_data(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + rng.normal(0, 0.5, 50)
        result = bagging(x, y, n_bags=30)
        assert isinstance(result, DescriptiveResult)
        preds = result.extra["predictions"]
        mse = np.mean((preds - (2 * x + 1)) ** 2)
        assert mse < 5.0

    def test_r_squared(self):
        x = np.linspace(0, 5, 30)
        y = 3 * x
        result = bagging(x, y, n_bags=20)
        assert result.extra["r_squared"] > 0.9
