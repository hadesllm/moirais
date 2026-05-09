"""Tests for moirais.fn.mginv -- Moore-Penrose pseudoinverse."""

import numpy as np
from moirais.fn.mginv import pseudoinverse, mginv
from moirais.fn._containers import DescriptiveResult


class TestMginv:
    def test_alias(self):
        assert mginv is pseudoinverse

    def test_inverse(self):
        A = np.array([[1, 0], [0, 2]], dtype=float)
        r = pseudoinverse(A)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(A @ r.extra["A_pinv"], np.eye(2), atol=1e-10)

    def test_rank_deficient(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        r = pseudoinverse(A)
        assert r.value == 1
        np.testing.assert_allclose(A @ r.extra["A_pinv"] @ A, A, atol=1e-10)

    def test_rectangular(self):
        A = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        r = pseudoinverse(A)
        assert r.value == 2
