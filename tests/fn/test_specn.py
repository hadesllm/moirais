"""Tests for moirais.fn.specn -- Periodogram."""
import numpy as np
import pytest
from moirais.fn.specn import periodogram


class TestPeriodogram:
    def test_basic(self):
        t = np.arange(100)
        y = np.sin(2 * np.pi * t / 10) + 0.1 * np.random.default_rng(42).standard_normal(100)
        res = periodogram(y)
        assert res.extra["peak_period"] > 0

    def test_short_raises(self):
        with pytest.raises(ValueError):
            periodogram(np.ones(2))

    def test_cheatsheet(self):
        from moirais.fn.specn import cheatsheet
        assert isinstance(cheatsheet(), str)
