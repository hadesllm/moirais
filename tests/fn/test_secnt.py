"""Tests for moirais.fn.secnt -- Secant method."""

import numpy as np
from moirais.fn.secnt import secant_method, secnt
from moirais.fn._containers import DescriptiveResult


class TestSecnt:
    def test_alias(self):
        assert secnt is secant_method

    def test_sqrt2(self):
        r = secant_method(lambda x: x**2 - 2, 1.0, 2.0)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - np.sqrt(2)) < 1e-8
        assert r.extra["converged"]

    def test_linear(self):
        r = secant_method(lambda x: 2 * x - 6, 0.0, 5.0)
        assert abs(r.value - 3.0) < 1e-10
