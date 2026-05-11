"""Tests for morie.fn.gseid -- Gauss-Seidel solver."""

import numpy as np
from morie.fn.gseid import gauss_seidel, gseid
from morie.fn._containers import DescriptiveResult


class TestGseid:
    def test_alias(self):
        assert gseid is gauss_seidel

    def test_solve(self):
        A = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        r = gauss_seidel(A, b)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-5)

    def test_faster_than_jacobi(self):
        from morie.fn.jacbi import jacobi_solve
        A = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        rg = gauss_seidel(A, b)
        rj = jacobi_solve(A, b)
        assert rg.extra["iterations"] <= rj.extra["iterations"]
