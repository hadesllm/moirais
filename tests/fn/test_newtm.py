"""Tests for moirais.fn.newtm -- Newton's method."""

import numpy as np
from moirais.fn.newtm import newton_method, newtm
from moirais.fn._containers import DescriptiveResult


class TestNewtm:
    def test_alias(self):
        assert newtm is newton_method

    def test_sqrt2(self):
        r = newton_method(lambda x: x**2 - 2, 1.5)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - np.sqrt(2)) < 1e-8
        assert r.extra["converged"]

    def test_with_derivative(self):
        r = newton_method(lambda x: x**3 - 1, 0.5, fprime=lambda x: 3 * x**2)
        assert abs(r.value - 1.0) < 1e-8

    def test_multivariate(self):
        def f(x):
            return np.array([x[0]**2 + x[1] - 1, x[0] + x[1]**2 - 1])
        r = newton_method(f, np.array([0.5, 0.5]))
        root = r.value
        np.testing.assert_allclose(f(root), [0, 0], atol=1e-6)
