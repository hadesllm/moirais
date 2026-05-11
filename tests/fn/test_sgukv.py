"""Tests for universal kriging variance."""
import numpy as np
from morie.fn.sgukv import sgukv


def test_sgukv_smoke():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    r = sgukv(coords, np.array([0.5, 0.5]))
    assert r.name == "universal_kriging_variance"
    assert r.statistic >= 0
    assert "trend_correction" in r.extra


def test_cheatsheet():
    from morie.fn.sgukv import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
