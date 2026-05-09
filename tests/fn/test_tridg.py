"""Tests for moirais.fn.tridg -- Thomas algorithm."""

import numpy as np
from moirais.fn.tridg import thomas_solve, tridg
from moirais.fn._containers import DescriptiveResult


class TestTridg:
    def test_alias(self):
        assert tridg is thomas_solve

    def test_simple(self):
        a = np.array([1, 1], dtype=float)
        b = np.array([4, 4, 4], dtype=float)
        c = np.array([1, 1], dtype=float)
        d = np.array([1, 2, 3], dtype=float)
        r = thomas_solve(a, b, c, d)
        assert isinstance(r, DescriptiveResult)
        A = np.diag(b) + np.diag(a, -1) + np.diag(c, 1)
        np.testing.assert_allclose(A @ r.extra["x"], d, atol=1e-10)

    def test_two_by_two(self):
        a = np.array([1.0])
        b = np.array([2.0, 3.0])
        c = np.array([1.0])
        d = np.array([5.0, 7.0])
        r = thomas_solve(a, b, c, d)
        np.testing.assert_allclose(r.extra["x"], np.linalg.solve(
            np.array([[2, 1], [1, 3]], dtype=float), d), atol=1e-10)
