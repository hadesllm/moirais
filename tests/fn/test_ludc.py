"""Tests for morie.fn.ludc -- LU decomposition."""

import numpy as np
from morie.fn.ludc import lu_decomposition, ludc
from morie.fn._containers import DescriptiveResult


class TestLudc:
    def test_alias(self):
        assert ludc is lu_decomposition

    def test_3x3(self):
        A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
        r = lu_decomposition(A)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["P"] @ A, r.extra["L"] @ r.extra["U"], atol=1e-10)

    def test_determinant(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        r = lu_decomposition(A)
        assert abs(r.value - np.linalg.det(A)) < 1e-8

    def test_identity(self):
        A = np.eye(3)
        r = lu_decomposition(A)
        np.testing.assert_allclose(r.extra["L"], np.eye(3), atol=1e-12)

    def test_non_square_raises(self):
        import pytest
        with pytest.raises(ValueError):
            lu_decomposition(np.ones((2, 3)))
