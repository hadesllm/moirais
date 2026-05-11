"""Tests for morie.fn.ncoef — normalize coefficients."""
import numpy as np
from morie.fn.ncoef import ncoef


def test_ncoef_smoke():
    r = ncoef([3.0, 4.0])
    assert r.name == "normalize_coefficients"
    assert abs(r.value - 5.0) < 1e-10
    unit = r.extra["unit_vector"]
    assert abs(np.linalg.norm(unit) - 1.0) < 1e-10


def test_cheatsheet():
    from morie.fn.ncoef import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
