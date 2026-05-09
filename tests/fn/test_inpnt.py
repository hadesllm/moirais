"""Tests for moirais.fn.inpnt -- Interior point method."""

import numpy as np
from moirais.fn.inpnt import interior_point_lp, inpnt
from moirais.fn._containers import DescriptiveResult


class TestInpnt:
    def test_alias(self):
        assert inpnt is interior_point_lp

    def test_returns_result(self):
        c = np.array([-1, 0], dtype=float)
        A_eq = np.array([[1, 1]], dtype=float)
        b_eq = np.array([3], dtype=float)
        r = interior_point_lp(c, A_eq, b_eq)
        assert isinstance(r, DescriptiveResult)
        assert "x" in r.extra
        assert "iterations" in r.extra

    def test_well_conditioned(self):
        c = np.array([1, 1, 0, 0], dtype=float)
        A_eq = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
        b_eq = np.array([2, 2], dtype=float)
        r = interior_point_lp(c, A_eq, b_eq)
        assert isinstance(r, DescriptiveResult)
