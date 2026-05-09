"""Tests for moirais.fn.sarma — seasonal ARMA."""
import numpy as np
import pytest
from moirais.fn.sarma import seasonal_arma


class TestSeasonalARMA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(120)
        for t in range(12, 120):
            y[t] += 0.5 * y[t-12]
        res = seasonal_arma(y, order=(0, 0), seasonal_order=(1, 0, 12))
        assert "sar" in res.extra

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            seasonal_arma(np.ones(5), seasonal_order=(1, 0, 12))
