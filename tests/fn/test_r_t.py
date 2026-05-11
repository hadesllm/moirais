"""Tests for morie.fn.r_t — Real-time Rt estimation."""

import numpy as np
import pytest

from morie.fn.r_t import realtime_rt


class TestRealtimeRt:
    def test_exponential_growth(self):
        rng = np.random.default_rng(42)
        inc = np.round(10 * np.exp(0.05 * np.arange(60)) + rng.poisson(2, 60))
        res = realtime_rt(inc, serial_mean=5.0, serial_sd=2.0)
        assert res.value > 1.0

    def test_declining_epidemic(self):
        inc = np.round(100 * np.exp(-0.05 * np.arange(60)) + 1)
        res = realtime_rt(inc, serial_mean=5.0, serial_sd=2.0)
        assert res.value < 1.5

    def test_short_series(self):
        with pytest.raises(ValueError):
            realtime_rt([1, 2, 3])
