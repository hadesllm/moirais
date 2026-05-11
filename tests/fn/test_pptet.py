"""Tests for morie.fn.pptet -- Phillips-Perron test."""
import numpy as np
import pytest
from morie.fn.pptet import pp_test


class TestPP:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)
        res = pp_test(y)
        assert "statistic" in res.extra

    def test_short_raises(self):
        with pytest.raises(ValueError):
            pp_test(np.ones(5))

    def test_cheatsheet(self):
        from morie.fn.pptet import cheatsheet
        assert isinstance(cheatsheet(), str)
