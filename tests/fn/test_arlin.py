"""Tests for arlin (areal interpolation)."""
import numpy as np
from moirais.fn.arlin import areal_interpolation


def test_areal_interpolation_basic():
    src = np.array([100.0, 200.0])
    areas = np.array([10.0, 20.0])
    overlap = np.array([[5.0, 5.0], [10.0, 10.0]])
    r = areal_interpolation(src, areas, overlap)
    assert r.extra is not None
    vals = np.array(r.extra["target_values"])
    assert len(vals) == 2
    assert np.all(vals >= 0)


def test_cheatsheet():
    from moirais.fn.arlin import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
