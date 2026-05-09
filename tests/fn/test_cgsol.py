"""Tests for moirais.fn.cgsol -- Conjugate gradient solver."""

import numpy as np
from moirais.fn.cgsol import conjugate_gradient, cgsol
from moirais.fn._containers import DescriptiveResult


class TestCgsol:
    def test_alias(self):
        assert cgsol is conjugate_gradient

    def test_spd_solve(self):
        A = np.array([[4, 1], [1, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        r = conjugate_gradient(A, b)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-6)

    def test_identity(self):
        b = np.array([1.0, 2.0, 3.0])
        r = conjugate_gradient(np.eye(3), b)
        np.testing.assert_allclose(r.extra["x"], b, atol=1e-10)
