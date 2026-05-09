"""Tests for moirais.fn.arima — ARIMA fitting."""
import numpy as np
import pytest
from moirais.fn.arima import arima_fit


class TestARIMA:
    def test_ar1(self):
        rng = np.random.default_rng(42)
        y = np.zeros(200)
        for t in range(1, 200):
            y[t] = 0.7 * y[t-1] + rng.standard_normal()
        res = arima_fit(y, order=(1, 0, 0))
        assert abs(res.extra["ar"][0] - 0.7) < 0.2

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            arima_fit(np.array([1.0, 2.0]), order=(3, 0, 0))
