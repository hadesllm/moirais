"""Tests for indicator kriging."""
import numpy as np
from morie.fn.sgind import sgind


def test_sgind_smoke():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    Z = np.array([0.5, 1.5, 0.3, 2.0])
    r = sgind(Z, coords, np.array([0.5, 0.5]), threshold=1.0)
    assert r.name == "indicator_kriging"
    assert 0.0 <= r.statistic <= 1.0
    assert "n_below" in r.extra


def test_cheatsheet():
    from morie.fn.sgind import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
