"""Tests for Matern variogram."""
import numpy as np
from moirais.fn.sgmat import sgmat


def test_sgmat_smoke():
    h = np.array([0, 0.5, 1, 2, 5], dtype=float)
    r = sgmat(h, nugget=0.0, sill=1.0, range_param=2.0, nu=1.5)
    assert r.name == "matern_variogram"
    assert "gamma" in r.extra
    assert r.extra["nu"] == 1.5


def test_sgmat_zero_lag():
    h = np.array([0.0])
    r = sgmat(h, nugget=0.0, sill=1.0, range_param=2.0)
    assert r.extra["gamma"][0] == 0.0
