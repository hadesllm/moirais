"""Tests for moirais.fn.hddif -- Hadamard differentiability check."""

import numpy as np
import pytest
from moirais.fn.hddif import hadamard_differentiability


class TestHadamardDifferentiability:
    def test_linear_functional(self):
        phi = lambda x: np.sum(x)
        theta = np.array([1.0, 2.0, 3.0])
        r = hadamard_differentiability(phi, theta)
        assert r["is_differentiable"]
        assert len(r["quotients"]) == 5

    def test_quadratic_functional(self):
        phi = lambda x: np.sum(x**2)
        theta = np.array([1.0, 2.0])
        r = hadamard_differentiability(phi, theta, tol=1e-2)
        assert r["is_differentiable"]

    def test_non_differentiable_at_zero(self):
        phi = lambda x: np.sum(np.abs(x))
        theta = np.zeros(3)
        h = np.array([1.0, -1.0, 0.5])
        r = hadamard_differentiability(phi, theta, h=h)
        assert r["derivative_estimate"] != 0

    def test_custom_t_values(self):
        phi = lambda x: np.mean(x)
        theta = np.array([1.0, 2.0])
        r = hadamard_differentiability(phi, theta, t_values=np.array([0.1, 0.01]))
        assert len(r["quotients"]) == 2

    def test_empty_theta_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            hadamard_differentiability(lambda x: 0, np.array([]))
