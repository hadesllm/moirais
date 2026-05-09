"""Tests for simple kriging variance."""
import numpy as np
from moirais.fn.sgskv import sgskv


def test_sgskv_smoke():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    r = sgskv(coords, np.array([0.5, 0.5]))
    assert r.name == "simple_kriging_variance"
    assert r.statistic >= 0


def test_sgskv_at_obs():
    coords = np.array([[0, 0], [1, 0]], dtype=float)
    r = sgskv(coords, np.array([0.0, 0.0]))
    assert r.statistic >= 0
