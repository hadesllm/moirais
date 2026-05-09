"""Tests for point_biserial."""
import numpy as np, pytest
from moirais.fn.pbis import point_biserial

class TestPBis:
    def test_correlated(self):
        binary = np.array([0]*20 + [1]*20, dtype=float)
        cont = np.concatenate([np.random.default_rng(0).normal(0,1,20), np.random.default_rng(0).normal(3,1,20)])
        r = point_biserial(binary, cont)
        assert r.estimate > 0.3

    def test_no_corr(self):
        rng = np.random.default_rng(1)
        binary = rng.integers(0, 2, 50).astype(float)
        cont = rng.normal(0, 1, 50)
        r = point_biserial(binary, cont)
        assert abs(r.estimate) < 0.5
