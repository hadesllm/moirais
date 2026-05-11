"""Tests for morie.fn.eig_ — eigenvalue analysis."""
import numpy as np
import pytest
from morie.fn.eig_ import eigen_analysis


class TestEigen:
    def test_symmetric(self):
        A = np.array([[2, 1], [1, 2]], dtype=float)
        res = eigen_analysis(A)
        assert res.extra["spectral_radius"] == pytest.approx(3.0, abs=1e-10)

    def test_trace(self):
        A = np.diag([1.0, 2.0, 3.0])
        res = eigen_analysis(A)
        assert res.extra["trace"] == pytest.approx(6.0, abs=1e-10)
