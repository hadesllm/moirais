"""Tests for moirais.fn.svdcp -- SVD computation."""

import numpy as np
from moirais.fn.svdcp import svd_compute, svdcp
from moirais.fn._containers import DescriptiveResult


class TestSvdcp:
    def test_alias(self):
        assert svdcp is svd_compute

    def test_rank(self):
        A = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        r = svd_compute(A)
        assert r.value == 2

    def test_reconstruction(self):
        A = np.random.default_rng(42).standard_normal((3, 3))
        r = svd_compute(A)
        U, S, Vt = r.extra["U"], r.extra["S"], r.extra["Vt"]
        np.testing.assert_allclose(U @ np.diag(S) @ Vt, A, atol=1e-10)

    def test_rank_deficient(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        r = svd_compute(A)
        assert r.value == 1
