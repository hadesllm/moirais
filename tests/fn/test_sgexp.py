"""Tests for exponential variogram."""
import numpy as np
from moirais.fn.sgexp import sgexp


def test_sgexp_smoke():
    h = np.array([0, 1, 2, 3, 5, 10], dtype=float)
    r = sgexp(h, nugget=0.0, sill=1.0, range_param=2.0)
    assert r.name == "exponential_variogram"
    assert r.extra["gamma"][0] == 0.0
    assert r.extra["practical_range"] == 6.0


def test_sgexp_approaches_sill():
    h = np.array([100.0])
    r = sgexp(h, nugget=0.0, sill=1.0, range_param=1.0)
    assert abs(r.extra["gamma"][0] - 1.0) < 1e-10
