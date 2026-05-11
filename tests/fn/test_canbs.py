"""Tests for canbs (Canberra distance)."""
import numpy as np
from morie.fn.canbs import canberra_dist


def test_canberra_identical():
    a = np.array([1.0, 2.0, 3.0])
    r = canberra_dist(a, a)
    assert abs(r.value) < 1e-10


def test_canberra_known():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    r = canberra_dist(a, b)
    assert abs(r.value - 2.0) < 1e-10
