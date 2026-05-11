"""Test gram_schmidt_orth."""
import numpy as np
import pytest
from morie.fn.gso import gram_schmidt_orth
from morie.fn._containers import DescriptiveResult


class TestGramSchmidtOrth:
    def test_basic(self):
        basis = np.array([[1, 1], [0, 1]], dtype=float)
        result = gram_schmidt_orth(basis=basis)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        basis = np.array([[1, 1], [0, 1]], dtype=float)
        result = gram_schmidt_orth(basis=basis)
        assert "orthogonal_basis" in result.extra

    def test_orthogonality(self):
        basis = np.array([[1, 1], [0, 1]], dtype=float)
        result = gram_schmidt_orth(basis=basis)
        ob = np.asarray(result.extra["orthogonal_basis"])
        dot = np.dot(ob[0], ob[1])
        assert abs(dot) < 1e-10
