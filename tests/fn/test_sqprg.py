"""Tests for moirais.fn.sqprg -- SQP optimizer."""

import numpy as np
from moirais.fn.sqprg import sqp_optimize, sqprg
from moirais.fn._containers import DescriptiveResult


class TestSqprg:
    def test_alias(self):
        assert sqprg is sqp_optimize

    def test_equality_constrained(self):
        f = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
        g = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2.5)])
        cons = [lambda x: x[0] + x[1] - 3]
        r = sqp_optimize(f, g, cons, np.array([0.0, 0.0]))
        assert isinstance(r, DescriptiveResult)
        assert abs(r.extra["x"][0] + r.extra["x"][1] - 3.0) < 0.1
