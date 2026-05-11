"""Tests for morie.fn.rmspd -- RMSProp optimizer."""

import numpy as np
from morie.fn.rmspd import rmsprop_optimize, rmspd
from morie.fn._containers import DescriptiveResult


class TestRmspd:
    def test_alias(self):
        assert rmspd is rmsprop_optimize

    def test_quadratic(self):
        f = lambda x: np.sum(x**2)
        g = lambda x: 2 * x
        r = rmsprop_optimize(f, g, np.array([5.0, 5.0]), lr=0.01, maxiter=3000)
        assert isinstance(r, DescriptiveResult)
        assert r.value < 1.0

    def test_converges(self):
        f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
        g = lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])
        r = rmsprop_optimize(f, g, np.array([0.0, 0.0]), lr=0.01)
        assert r.value < 0.1
