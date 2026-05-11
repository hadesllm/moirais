"""Tests for morie.fn.cgd — conjugate gradient descent."""
import numpy as np
import pytest
from morie.fn.cgd import conjugate_gradient


class TestConjugateGradient:
    def test_returns_result(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad_f = lambda x: np.array([2 * x[0], 2 * x[1]])
        x0 = np.array([5.0, -3.0])
        res = conjugate_gradient(f, grad_f, x0)
        assert "x_opt" in res.extra
        assert "f_opt" in res.extra
        assert isinstance(res.extra["f_opt"], float)

    def test_f_opt_leq_initial(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad_f = lambda x: np.array([2 * x[0], 2 * x[1]])
        x0 = np.array([1.0, 1.0])
        res = conjugate_gradient(f, grad_f, x0)
        assert res.extra["f_opt"] <= f(x0) + 1e-10

    def test_n_iter_positive(self):
        f = lambda x: (x[0] - 1) ** 2 + (x[1] + 2) ** 2
        grad_f = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] + 2)])
        x0 = np.array([0.0, 0.0])
        res = conjugate_gradient(f, grad_f, x0)
        assert res.extra["n_iter"] > 0
        assert isinstance(res.extra["final_grad_norm"], float)
