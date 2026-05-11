"""Tests for morie.fn.qrdcp -- QR decomposition."""

import numpy as np
from morie.fn.qrdcp import qr_decomposition, qrdcp
from morie.fn._containers import DescriptiveResult


class TestQrdcp:
    def test_alias(self):
        assert qrdcp is qr_decomposition

    def test_reconstruction(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        r = qr_decomposition(A)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["Q"] @ r.extra["R"], A, atol=1e-10)

    def test_orthogonality(self):
        A = np.random.default_rng(42).standard_normal((4, 3))
        r = qr_decomposition(A)
        Q = r.extra["Q"]
        np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-10)

    def test_square(self):
        A = np.array([[1, 0], [0, 1]], dtype=float)
        r = qr_decomposition(A)
        np.testing.assert_allclose(r.extra["Q"] @ r.extra["R"], A, atol=1e-12)
