"""Tests for moirais.fn.spdmt -- Sparse diagonal matrix operations."""

import numpy as np
from moirais.fn.spdmt import sparse_diagonal, spdmt
from moirais.fn._containers import DescriptiveResult


class TestSpdmt:
    def test_alias(self):
        assert spdmt is sparse_diagonal

    def test_main_diag(self):
        r = sparse_diagonal([np.array([1, 2, 3])], offsets=[0])
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["matrix"], np.diag([1, 2, 3]))
        assert r.value == 3

    def test_tridiagonal(self):
        main = np.array([4, 4, 4])
        sub = np.array([1, 1])
        sup = np.array([1, 1])
        r = sparse_diagonal([sub, main, sup], offsets=[-1, 0, 1], n=3)
        M = r.extra["matrix"]
        assert M[0, 0] == 4
        assert M[1, 0] == 1
        assert M[0, 1] == 1
