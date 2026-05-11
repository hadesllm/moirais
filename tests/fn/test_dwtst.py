"""Tests for morie.fn.dwtst -- Durbin-Watson statistic."""
import numpy as np
import pytest
from morie.fn.dwtst import durbin_watson


class TestDW:
    def test_no_autocorr(self):
        rng = np.random.default_rng(42)
        e = rng.standard_normal(100)
        res = durbin_watson(e)
        assert 1.0 < res.value < 3.0

    def test_short_raises(self):
        with pytest.raises(ValueError):
            durbin_watson(np.ones(2))

    def test_cheatsheet(self):
        from morie.fn.dwtst import cheatsheet
        assert isinstance(cheatsheet(), str)
