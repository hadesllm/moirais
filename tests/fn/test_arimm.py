"""Tests for morie.fn.arimm -- ARIMA(p,d,q)."""
import numpy as np
import pytest
from morie.fn.arimm import arima_fit


class TestARIMAFit:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100))
        res = arima_fit(y, p=1, d=1, q=0)
        assert res.name == "arima_fit"
        assert "aic" in res.extra

    def test_short_raises(self):
        with pytest.raises(ValueError):
            arima_fit(np.ones(5), p=1, d=1)

    def test_cheatsheet(self):
        from morie.fn.arimm import cheatsheet
        assert isinstance(cheatsheet(), str)
