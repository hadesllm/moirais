"""Tests for moirais.fn.megtr -- Kronecker product decomposition."""

import numpy as np
from moirais.fn.megtr import kronecker_decompose, megtr
from moirais.fn._containers import DescriptiveResult


class TestMegtr:
    def test_alias(self):
        assert megtr is kronecker_decompose

    def test_exact_kron(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        C = np.kron(A, B)
        r = kronecker_decompose(C, m1=2, n1=2)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["rel_error"] < 0.01

    def test_returns_factors(self):
        C = np.random.default_rng(42).normal(0, 1, (4, 4))
        r = kronecker_decompose(C, m1=2, n1=2)
        assert "A" in r.value and "B" in r.value
