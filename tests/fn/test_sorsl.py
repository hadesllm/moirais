"""Tests for moirais.fn.sorsl -- SOR solver."""

import numpy as np
from moirais.fn.sorsl import sor_solve, sorsl
from moirais.fn._containers import DescriptiveResult


class TestSorsl:
    def test_alias(self):
        assert sorsl is sor_solve

    def test_solve(self):
        A = np.array([[4, 1], [1, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        r = sor_solve(A, b, omega=1.2)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-5)
        assert r.extra["omega"] == 1.2

    def test_omega_1_like_gauss_seidel(self):
        A = np.array([[4, 1], [1, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        r = sor_solve(A, b, omega=1.0)
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-5)
