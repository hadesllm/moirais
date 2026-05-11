"""Tests for morie.fn.chles -- Cholesky decomposition."""

import numpy as np
import pytest
from morie.fn.chles import cholesky_solve, chles
from morie.fn._containers import DescriptiveResult


class TestChles:
    def test_alias(self):
        assert chles is cholesky_solve

    def test_decomposition(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        r = cholesky_solve(A)
        assert isinstance(r, DescriptiveResult)
        L = r.extra["L"]
        np.testing.assert_allclose(L @ L.T, A, atol=1e-10)

    def test_solve(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        r = cholesky_solve(A, b)
        x = r.extra["x"]
        np.testing.assert_allclose(A @ x, b, atol=1e-10)

    def test_not_pd_raises(self):
        A = np.array([[-1, 0], [0, 1]], dtype=float)
        with pytest.raises(ValueError):
            cholesky_solve(A)
