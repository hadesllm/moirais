"""Tests for morie.fn.lowes -- LOWESS smoother."""

import numpy as np
from morie.fn.lowes import lowess, lowes
from morie.fn._containers import DescriptiveResult


class TestLowes:
    def test_alias(self):
        assert lowes is lowess

    def test_noisy_linear(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y_true = 2 * x + 1
        y = y_true + rng.normal(0, 1, 50)
        result = lowess(x, y, frac=0.3)
        assert isinstance(result, DescriptiveResult)
        smoothed = result.value
        mse_raw = np.mean((y - y_true) ** 2)
        mse_smooth = np.mean((smoothed - y_true) ** 2)
        assert mse_smooth < mse_raw

    def test_output_length(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lowess(x, y)
        assert len(result.value) == 5
