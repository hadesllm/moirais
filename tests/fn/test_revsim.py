"""Tests for morie.fn.revsim -- constrained LP optimization."""

import numpy as np
from morie.fn.revsim import revised_simplex, revsim
from morie.fn._containers import DescriptiveResult


class TestRevsim:
    def test_alias(self):
        assert revsim is revised_simplex

    def test_simple_lp(self):
        c = np.array([-1, -2])
        A_ub = np.array([[1, 1], [2, 1]])
        b_ub = np.array([4, 6])
        r = revised_simplex(c, A_ub, b_ub)
        assert isinstance(r, DescriptiveResult)
        assert r.value["success"]
        assert r.value["fun"] < 0

    def test_returns_dict(self):
        c = np.array([1, 1])
        r = revised_simplex(c)
        assert "x" in r.value
