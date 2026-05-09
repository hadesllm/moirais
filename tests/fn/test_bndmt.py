"""Tests for moirais.fn.bndmt -- Banded matrix solver."""

import numpy as np
from moirais.fn.bndmt import banded_solve, bndmt
from moirais.fn._containers import DescriptiveResult


class TestBndmt:
    def test_alias(self):
        assert bndmt is banded_solve

    def test_tridiag(self):
        A = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        r = banded_solve(A, b)
        assert isinstance(r, DescriptiveResult)
        assert r.value == 1
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-10)

    def test_diagonal(self):
        A = np.diag([2.0, 3.0, 4.0])
        b = np.array([2, 6, 12], dtype=float)
        r = banded_solve(A, b)
        assert r.value == 0
        np.testing.assert_allclose(r.extra["x"], [1, 2, 3], atol=1e-10)
