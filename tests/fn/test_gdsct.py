"""Tests for moirais.fn.gdsct -- Gradient descent."""

import numpy as np
from moirais.fn.gdsct import gradient_descent, gdsct
from moirais.fn._containers import DescriptiveResult


class TestGdsct:
    def test_alias(self):
        assert gdsct is gradient_descent

    def test_quadratic(self):
        f = lambda x: np.sum(x**2)
        g = lambda x: 2 * x
        r = gradient_descent(f, g, np.array([5.0, 5.0]), lr=0.1)
        assert isinstance(r, DescriptiveResult)
        assert r.value < 1e-6
        assert r.extra["converged"]

    def test_momentum(self):
        f = lambda x: np.sum(x**2)
        g = lambda x: 2 * x
        r = gradient_descent(f, g, np.array([10.0]), lr=0.01, momentum=0.9)
        assert r.value < 1e-4
