"""Tests for semipartial_corr."""
import numpy as np, pytest
from morie.fn.spcor import semipartial_corr

class TestSpcor:
    def test_basic(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, 50)
        x = z + rng.normal(0, 0.5, 50)
        y = z + rng.normal(0, 0.5, 50)
        r = semipartial_corr(x, y, z)
        assert r.measure == "semipartial_r"

    def test_has_pvalue(self):
        rng = np.random.default_rng(1)
        r = semipartial_corr(rng.normal(0,1,30), rng.normal(0,1,30), rng.normal(0,1,30))
        assert "p_value" in r.extra
