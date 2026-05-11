"""Tests for morie.fn.dst2s -- distance to squared."""

import numpy as np
from morie.fn.dst2s import distance_to_squared, dst2s


def test_dst2s_smoke():
    D = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]], dtype=float)
    r = dst2s(D)
    assert r.name == "distance_to_squared"
    assert r.value.shape == (3, 3)
    assert r.value[0, 1] == 4.0
    assert r.value[0, 2] == 9.0


def test_dst2s_alias():
    assert dst2s is distance_to_squared


def test_dst2s_zeros():
    D = np.zeros((4, 4))
    r = dst2s(D)
    assert np.all(r.value == 0)
