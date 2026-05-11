"""Tests for spherical variogram."""
import numpy as np
from morie.fn.sgsph import sgsph


def test_sgsph_smoke():
    h = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    r = sgsph(h, nugget=0.1, sill=1.0, range_param=3.0)
    assert r.name == "spherical_variogram"
    assert "gamma" in r.extra
    assert r.extra["gamma"][0] == 0.0


def test_sgsph_at_range():
    h = np.array([3.0])
    r = sgsph(h, nugget=0.0, sill=1.0, range_param=3.0)
    assert abs(r.extra["gamma"][0] - 1.0) < 1e-10


def test_sgsph_beyond_range():
    h = np.array([5.0])
    r = sgsph(h, nugget=0.0, sill=1.0, range_param=3.0)
    assert abs(r.extra["gamma"][0] - 1.0) < 1e-10
