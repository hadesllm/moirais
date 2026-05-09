"""Tests for moirais.fn.arcit -- Gauss-Seidel solver."""

import numpy as np
from moirais.fn.arcit import gauss_seidel, arcit
from moirais.fn._containers import DescriptiveResult


class TestArcit:
    def test_alias(self):
        assert arcit is gauss_seidel

    def test_simple_system(self):
        A = np.array([[4, 1], [1, 3]], dtype=float)
        b = np.array([1, 2], dtype=float)
        result = gauss_seidel(A, b)
        assert isinstance(result, DescriptiveResult)
        x = np.array(result.value)
        assert np.allclose(A @ x, b, atol=1e-6)
        assert result.extra["converged"]

    def test_identity(self):
        A = np.eye(3)
        b = np.array([1, 2, 3], dtype=float)
        result = gauss_seidel(A, b)
        assert np.allclose(result.value, [1, 2, 3], atol=1e-6)
