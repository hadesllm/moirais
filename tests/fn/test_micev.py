"""Tests for mice_impute."""
import numpy as np, pytest
from morie.fn.micev import mice_impute

class TestMICE:
    def test_basic(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (30, 3))
        data[0, 0] = np.nan
        data[5, 1] = np.nan
        r = mice_impute(data, n_imputations=3, seed=0)
        assert r.extra["n_missing"] == 2

    def test_no_missing(self):
        data = np.ones((10, 2))
        r = mice_impute(data, seed=0)
        assert r.value == 0
