"""Tests for moirais.fn.jacbi -- Jacobi solver."""

import numpy as np
import pytest
from moirais.fn.jacbi import jacobi_solve, jacbi
from moirais.fn._containers import DescriptiveResult


class TestJacbi:
    def test_alias(self):
        assert jacbi is jacobi_solve

    def test_diag_dominant(self):
        A = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        r = jacobi_solve(A, b)
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(A @ r.extra["x"], b, atol=1e-5)

    def test_zero_diag_raises(self):
        A = np.array([[0, 1], [1, 1]], dtype=float)
        with pytest.raises(ValueError):
            jacobi_solve(A, np.array([1, 1]))
