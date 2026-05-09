"""Tests for moirais.fn.ljbxt -- Ljung-Box test."""
import numpy as np
import pytest
from moirais.fn.ljbxt import ljung_box


class TestLjungBox:
    def test_white_noise(self):
        rng = np.random.default_rng(42)
        e = rng.standard_normal(200)
        res = ljung_box(e, lags=10)
        assert res.extra["p_value"] > 0.01

    def test_correlated(self):
        rng = np.random.default_rng(42)
        e = np.zeros(200)
        e[0] = rng.standard_normal()
        for t in range(1, 200):
            e[t] = 0.8 * e[t - 1] + rng.standard_normal()
        res = ljung_box(e, lags=10)
        assert res.extra["Q"] > 0

    def test_cheatsheet(self):
        from moirais.fn.ljbxt import cheatsheet
        assert isinstance(cheatsheet(), str)
