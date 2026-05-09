"""Tests for moirais.fn.adfrr -- Augmented Dickey-Fuller test."""
import numpy as np
import pytest
from moirais.fn.adfrr import adf_test


class TestADF:
    def test_stationary(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(200)
        res = adf_test(y)
        assert res.extra["statistic"] < 0

    def test_unit_root(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(200))
        res = adf_test(y)
        assert res.extra["statistic"] is not None

    def test_short_raises(self):
        with pytest.raises(ValueError):
            adf_test(np.ones(3))

    def test_cheatsheet(self):
        from moirais.fn.adfrr import cheatsheet
        assert isinstance(cheatsheet(), str)
