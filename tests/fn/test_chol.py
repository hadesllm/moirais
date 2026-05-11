"""Tests for morie.fn.chol — Cholesky decomposition."""
import numpy as np
import pytest
from morie.fn.chol import cholesky_decompose


class TestCholesky:
    def test_identity(self):
        res = cholesky_decompose(np.eye(3))
        np.testing.assert_allclose(res.extra["L"], np.eye(3))

    def test_reconstruction(self):
        A = np.array([[4, 2], [2, 3]], dtype=float)
        res = cholesky_decompose(A)
        L = res.extra["L"]
        np.testing.assert_allclose(L @ L.T, A, atol=1e-10)

    def test_not_pd_raises(self):
        with pytest.raises(ValueError):
            cholesky_decompose(np.array([[-1, 0], [0, -1]]))
