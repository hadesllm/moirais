"""Tests for morie.fn.nash — Nash equilibrium."""
import numpy as np
from morie.fn.nash import nash_equilibrium


class TestNash:
    def test_prisoners_dilemma(self):
        A = np.array([[-1, -3], [0, -2]])
        B = np.array([[-1, 0], [-3, -2]])
        res = nash_equilibrium(A, B)
        assert res.value >= 1

    def test_matching_pennies(self):
        A = np.array([[1, -1], [-1, 1]])
        B = np.array([[-1, 1], [1, -1]])
        res = nash_equilibrium(A, B)
        assert len(res.extra["equilibria"]) >= 1
