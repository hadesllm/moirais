"""Tests for moirais.fn.kron — Kronecker product."""
import numpy as np
from moirais.fn.kron import kronecker_product


class TestKronecker:
    def test_shape(self):
        A = np.eye(2)
        B = np.ones((3, 3))
        res = kronecker_product(A, B)
        assert res.extra["product"].shape == (6, 6)

    def test_identity_kron(self):
        A = np.array([[1, 2], [3, 4]])
        I = np.eye(2)
        res = kronecker_product(A, I)
        K = res.extra["product"]
        assert K.shape == (4, 4)
        expected = np.kron(A, I)
        np.testing.assert_allclose(K, expected)
