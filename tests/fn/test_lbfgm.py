"""Tests for morie.fn.lbfgm -- L-BFGS optimizer."""

import numpy as np
from morie.fn.lbfgm import lbfgs_optimize, lbfgm
from morie.fn._containers import DescriptiveResult


class TestLbfgm:
    def test_alias(self):
        assert lbfgm is lbfgs_optimize

    def test_quadratic(self):
        f = lambda x: np.sum(x**2)
        g = lambda x: 2 * x
        r = lbfgs_optimize(f, g, np.array([5.0, 5.0]))
        assert isinstance(r, DescriptiveResult)
        assert r.value < 1e-10
        assert r.extra["converged"]

    def test_rosenbrock(self):
        f = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        g = lambda x: np.array([
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
            200 * (x[1] - x[0]**2),
        ])
        r = lbfgs_optimize(f, g, np.array([0.0, 0.0]), maxiter=500)
        assert r.value < 1.0
