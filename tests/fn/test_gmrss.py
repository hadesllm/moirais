"""Tests for moirais.fn.gmrss -- GMRES solver."""

import numpy as np
from moirais.fn.gmrss import gmres_solve, gmrss
from moirais.fn._containers import DescriptiveResult


class TestGmrss:
    def test_alias(self):
        assert gmrss is gmres_solve

    def test_solve(self):
        A = np.array([[4, 1], [1, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        r = gmres_solve(A, b)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-6)

    def test_convergence(self):
        n = 10
        A = np.eye(n) * 4 + np.random.default_rng(42).standard_normal((n, n)) * 0.1
        b = np.ones(n)
        r = gmres_solve(A, b)
        assert r.value < 1e-6
