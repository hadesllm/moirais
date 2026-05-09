"""Tests for moirais.fn.unfld — Unfolding model."""
import numpy as np
from moirais.fn.unfld import unfld


def test_unfld_basic():
    R = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
    r = unfld(R, n_dims=1)
    assert r.coordinates.shape[0] == 6
    assert r.stress >= 0


def test_unfld_2d():
    R = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3]])
    r = unfld(R, n_dims=2)
    assert r.coordinates.shape == (7, 2)


def test_unfld_perfect():
    R = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    r = unfld(R, n_dims=1)
    assert r.stress < 5.0
