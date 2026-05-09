"""Tests for cheby (Chebyshev distance)."""
import numpy as np
from moirais.fn.cheby import chebyshev_dist


def test_chebyshev_basic():
    a = np.array([1.0, 5.0, 3.0])
    b = np.array([2.0, 1.0, 4.0])
    r = chebyshev_dist(a, b)
    assert abs(r.value - 4.0) < 1e-10


def test_chebyshev_identical():
    a = np.array([1.0, 2.0])
    r = chebyshev_dist(a, a)
    assert abs(r.value) < 1e-10
