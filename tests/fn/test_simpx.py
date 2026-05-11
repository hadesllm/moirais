"""Tests for morie.fn.simpx -- Simplex method."""

import numpy as np
from morie.fn.simpx import simplex_lp, simpx
from morie.fn._containers import DescriptiveResult


class TestSimpx:
    def test_alias(self):
        assert simpx is simplex_lp

    def test_simple_lp(self):
        c = np.array([-1, -2], dtype=float)
        A = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
        b = np.array([4, 3, 2], dtype=float)
        r = simplex_lp(c, A, b)
        assert isinstance(r, DescriptiveResult)
        assert r.value < 0
        x = r.extra["x"]
        assert np.all(A @ x <= b + 1e-6)

    def test_trivial(self):
        c = np.array([-1], dtype=float)
        A = np.array([[1]], dtype=float)
        b = np.array([5], dtype=float)
        r = simplex_lp(c, A, b)
        assert abs(r.extra["x"][0] - 5.0) < 1e-6
