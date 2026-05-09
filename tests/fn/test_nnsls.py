"""Tests for moirais.fn.nnsls -- Non-negative least squares."""

import numpy as np
from moirais.fn.nnsls import nnls, nnsls
from moirais.fn._containers import DescriptiveResult


class TestNnsls:
    def test_alias(self):
        assert nnsls is nnls

    def test_nonneg(self):
        A = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
        b = np.array([2, 3, 1], dtype=float)
        r = nnls(A, b)
        assert isinstance(r, DescriptiveResult)
        assert np.all(r.extra["x"] >= -1e-10)

    def test_forced_zero(self):
        A = np.array([[1, -1]], dtype=float).T
        b = np.array([1], dtype=float)
        r = nnls(A.T, b)
        assert r.extra["x"][0] >= 0
