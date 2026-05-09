"""Tests for moirais.fn.ironm -- constrained LP optimization."""

import numpy as np
from moirais.fn.ironm import armor_optimize, ironm
from moirais.fn._containers import DescriptiveResult


class TestIronm:
    def test_alias(self):
        assert ironm is armor_optimize

    def test_simple_lp(self):
        c = np.array([-1, -2])
        A_ub = np.array([[1, 1], [2, 1]])
        b_ub = np.array([4, 6])
        r = armor_optimize(c, A_ub, b_ub)
        assert isinstance(r, DescriptiveResult)
        assert r.value["success"]
        assert r.value["fun"] < 0

    def test_returns_dict(self):
        c = np.array([1, 1])
        r = armor_optimize(c)
        assert "x" in r.value
