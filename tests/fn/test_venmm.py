"""Tests for moirais.fn.venmm -- co-occurrence matrix analysis."""

import numpy as np
from moirais.fn.venmm import cooccurrence_matrix, venmm
from moirais.fn._containers import DescriptiveResult


class TestVenmm:
    def test_alias(self):
        assert venmm is cooccurrence_matrix

    def test_binary(self):
        X = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
        r = cooccurrence_matrix(X)
        assert isinstance(r, DescriptiveResult)
        C = r.value["cooccurrence"]
        assert C.shape == (3, 3)

    def test_jaccard_range(self):
        rng = np.random.default_rng(42)
        X = rng.binomial(1, 0.5, (20, 5))
        r = cooccurrence_matrix(X)
        J = r.value["jaccard"]
        assert np.all(J >= 0) and np.all(J <= 1)
