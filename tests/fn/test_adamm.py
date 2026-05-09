"""Tests for moirais.fn.adamm -- Adam optimizer."""

import numpy as np
from moirais.fn.adamm import adam_optimize, adamm
from moirais.fn._containers import DescriptiveResult


class TestAdamm:
    def test_alias(self):
        assert adamm is adam_optimize

    def test_rosenbrock_2d(self):
        f = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        g = lambda x: np.array([
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
            200 * (x[1] - x[0]**2),
        ])
        r = adam_optimize(f, g, np.array([0.0, 0.0]), lr=0.01, maxiter=5000)
        assert isinstance(r, DescriptiveResult)
        assert r.value < 1.0

    def test_quadratic(self):
        f = lambda x: np.sum(x**2)
        g = lambda x: 2 * x
        r = adam_optimize(f, g, np.array([5.0, 5.0]), lr=0.01, maxiter=5000)
        assert r.value < 50.0
        assert r.value < f(np.array([5.0, 5.0]))
