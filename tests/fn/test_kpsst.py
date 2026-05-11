"""Tests for morie.fn.kpsst -- KPSS stationarity test."""
import numpy as np
import pytest
from morie.fn.kpsst import kpss_test


class TestKPSS:
    def test_stationary(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        res = kpss_test(y)
        assert res.extra["statistic"] >= 0

    def test_trend(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100) + np.arange(100) * 0.01
        res = kpss_test(y, regression="ct")
        assert "10%" in res.extra["critical_values"]

    def test_short_raises(self):
        with pytest.raises(ValueError):
            kpss_test(np.ones(5))

    def test_cheatsheet(self):
        from morie.fn.kpsst import cheatsheet
        assert isinstance(cheatsheet(), str)
