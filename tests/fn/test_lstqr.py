"""Tests for morie.fn.lstqr -- Least squares via QR."""

import numpy as np
from morie.fn.lstqr import lstsq_qr, lstqr
from morie.fn._containers import DescriptiveResult


class TestLstqr:
    def test_alias(self):
        assert lstqr is lstsq_qr

    def test_exact_solve(self):
        A = np.array([[1, 1], [0, 1]], dtype=float)
        b = np.array([3, 2], dtype=float)
        r = lstsq_qr(A, b)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["x"], [1, 2], atol=1e-10)
        assert r.value < 1e-10

    def test_overdetermined(self):
        A = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
        b = np.array([3, 1, 2], dtype=float)
        r = lstsq_qr(A, b)
        assert len(r.extra["x"]) == 2
