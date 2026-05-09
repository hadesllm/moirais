"""Tests for moirais.fn.matnm -- Matrix norms."""

import numpy as np
from moirais.fn.matnm import matrix_norms, matnm
from moirais.fn._containers import DescriptiveResult


class TestMatnm:
    def test_alias(self):
        assert matnm is matrix_norms

    def test_identity(self):
        r = matrix_norms(np.eye(3))
        assert isinstance(r, DescriptiveResult)
        assert abs(r.extra["norm_1"] - 1.0) < 1e-12
        assert abs(r.extra["norm_2"] - 1.0) < 1e-12
        assert abs(r.extra["norm_inf"] - 1.0) < 1e-12
        assert abs(r.extra["norm_fro"] - np.sqrt(3)) < 1e-12

    def test_simple(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        r = matrix_norms(A)
        assert r.value > 0
        assert r.extra["norm_1"] == 6.0
