"""Tests for morie.fn.coitg -- Engle-Granger cointegration."""
import numpy as np
import pytest
from morie.fn.coitg import eg_coint


class TestEGCoint:
    def test_cointegrated(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(100))
        y = 2 * x + rng.normal(0, 0.5, 100)
        res = eg_coint(y, x)
        assert res.extra["adf_statistic"] < 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            eg_coint(np.ones(10), np.ones(15))

    def test_cheatsheet(self):
        from morie.fn.coitg import cheatsheet
        assert isinstance(cheatsheet(), str)
