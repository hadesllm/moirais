"""Tests for moirais.fn.hotrd -- Newton's method convergence."""

import numpy as np
from moirais.fn.hotrd import newton_convergence, hotrd
from moirais.fn._containers import DescriptiveResult


class TestHotrd:
    def test_alias(self):
        assert hotrd is newton_convergence

    def test_sqrt2(self):
        r = newton_convergence(
            f=lambda x: x**2 - 2,
            fprime=lambda x: 2*x,
            x0=1.5,
        )
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - np.sqrt(2)) < 1e-10

    def test_convergence_order(self):
        r = newton_convergence(
            f=lambda x: x**3 - 1,
            fprime=lambda x: 3*x**2,
            x0=2.0,
        )
        assert r.extra["n_iter"] < 20
