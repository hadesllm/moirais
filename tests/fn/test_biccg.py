"""Tests for moirais.fn.biccg -- BiCGSTAB solver."""

import numpy as np
from moirais.fn.biccg import bicgstab, biccg
from moirais.fn._containers import DescriptiveResult


class TestBiccg:
    def test_alias(self):
        assert biccg is bicgstab

    def test_solve(self):
        A = np.array([[4, 1], [2, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        r = bicgstab(A, b)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-5)

    def test_nonsymmetric(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5)) + 5 * np.eye(5)
        x_true = rng.standard_normal(5)
        b = A @ x_true
        r = bicgstab(A, b)
        np.testing.assert_allclose(r.extra["x"], x_true, atol=1e-4)
