"""Tests for morie.fn.sarim -- Seasonal ARIMA."""
import numpy as np
import pytest
from morie.fn.sarim import sarima_fit


class TestSARIMA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = np.sin(np.linspace(0, 8 * np.pi, 120)) + rng.normal(0, 0.1, 120)
        y = np.cumsum(y) * 0.01 + y
        res = sarima_fit(y, p=1, d=1, q=0, P=1, D=1, Q=0, m=12)
        assert res.name == "sarima_fit"
        assert "order" in res.extra

    def test_short_raises(self):
        with pytest.raises(ValueError):
            sarima_fit(np.ones(10), m=12)

    def test_cheatsheet(self):
        from morie.fn.sarim import cheatsheet
        assert isinstance(cheatsheet(), str)
