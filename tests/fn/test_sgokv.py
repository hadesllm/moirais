"""Tests for ordinary kriging variance."""
import numpy as np
from moirais.fn.sgokv import sgokv


def test_sgokv_smoke():
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    r = sgokv(coords, np.array([0.5, 0.5]))
    assert r.name == "ordinary_kriging_variance"
    assert r.statistic >= 0


def test_cheatsheet():
    from moirais.fn.sgokv import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
