"""Tests for morie.fn.grwth -- exponential growth rate."""

import numpy as np
import pytest
from morie.fn.grwth import exponential_growth_rate


class TestGrowthRate:
    def test_known_growth(self):
        r_true = 0.1
        t = np.arange(30)
        inc = 10 * np.exp(r_true * t)
        res = exponential_growth_rate(inc)
        assert res["growth_rate"] == pytest.approx(r_true, abs=0.01)

    def test_doubling_time(self):
        r_true = 0.1
        t = np.arange(30)
        inc = 10 * np.exp(r_true * t)
        res = exponential_growth_rate(inc)
        expected_dt = np.log(2) / r_true
        assert res["doubling_time"] == pytest.approx(expected_dt, rel=0.1)

    def test_r_squared_high(self):
        t = np.arange(20)
        inc = 5 * np.exp(0.05 * t)
        res = exponential_growth_rate(inc)
        assert res["r_squared"] > 0.99

    def test_window_param(self):
        t = np.arange(50)
        inc = 5 * np.exp(0.1 * t)
        res = exponential_growth_rate(inc, window=(5, 25))
        assert res["growth_rate"] == pytest.approx(0.1, abs=0.01)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            exponential_growth_rate(np.array([0, 0, 5]))

    def test_ci_contains_rate(self):
        rng = np.random.default_rng(42)
        t = np.arange(40)
        inc = 10 * np.exp(0.08 * t) * rng.lognormal(0, 0.1, 40)
        res = exponential_growth_rate(inc)
        assert res["ci_lower"] <= res["growth_rate"] <= res["ci_upper"]
