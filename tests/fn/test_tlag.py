"""Tests for moirais.fn.tlag — Time-lag cross-correlation."""

import numpy as np
import pytest

from moirais.fn.tlag import time_lag_analysis


class TestTimeLag:
    def test_zero_lag(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        res = time_lag_analysis(x, x, max_lag=5)
        assert res.extra["best_lag"] == 0
        assert res.extra["best_corr"] == pytest.approx(1.0, abs=0.01)

    def test_lagged_signal(self):
        x = np.zeros(50)
        x[10:20] = 1
        y = np.zeros(50)
        y[13:23] = 1
        res = time_lag_analysis(x, y, max_lag=10)
        assert res.extra["best_lag"] in range(-5, 6)

    def test_too_short(self):
        with pytest.raises(ValueError):
            time_lag_analysis([1, 2, 3], [1, 2, 3], max_lag=5)
