"""Tests for little_mcar_test."""
import numpy as np, pytest
from moirais.fn.ltmcr import little_mcar_test

class TestLittleMCAR:
    def test_mcar(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (50, 3))
        mask = rng.random((50, 3)) < 0.1
        data[mask] = np.nan
        r = little_mcar_test(data)
        assert r.test_name == "Little MCAR"
        assert 0 <= r.p_value <= 1

    def test_complete(self):
        data = np.ones((20, 3))
        r = little_mcar_test(data)
        assert r.extra["n_patterns"] == 1
