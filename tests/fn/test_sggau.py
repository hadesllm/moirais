"""Tests for Gaussian variogram."""
import numpy as np
from moirais.fn.sggau import sggau


def test_sggau_smoke():
    h = np.array([0, 1, 2, 3, 5], dtype=float)
    r = sggau(h, nugget=0.0, sill=1.0, range_param=2.0)
    assert r.name == "gaussian_variogram"
    assert r.extra["gamma"][0] == 0.0
    assert r.extra["model"] == "gaussian"


def test_sggau_approaches_sill():
    h = np.array([50.0])
    r = sggau(h, nugget=0.0, sill=1.0, range_param=1.0)
    assert abs(r.extra["gamma"][0] - 1.0) < 1e-10
