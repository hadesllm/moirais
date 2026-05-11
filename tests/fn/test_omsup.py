"""Tests for morie.fn.omsup -- minimax optimization."""

import numpy as np
from morie.fn.omsup import minimax_solve, omsup
from morie.fn._containers import DescriptiveResult


class TestOmsup:
    def test_alias(self):
        assert omsup is minimax_solve

    def test_matching_pennies(self):
        A = np.array([[1, -1], [-1, 1]], dtype=float)
        r = minimax_solve(A)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - 0.0) < 0.1

    def test_dominant_strategy(self):
        A = np.array([[3, 0], [5, 1]], dtype=float)
        r = minimax_solve(A)
        assert r.extra["row_strategy"] is not None
