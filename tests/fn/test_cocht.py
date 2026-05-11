"""Tests for morie.fn.cocht -- Cochran's Q test."""

import numpy as np
import pytest
from morie.fn.cocht import cochrans_q_test
from morie.fn._containers import TestResult


class TestCochranQ:
    def test_equal_proportions(self):
        """Identical columns => Q near 0, non-significant."""
        data = np.array([[1, 1, 1]] * 20 + [[0, 0, 0]] * 10)
        r = cochrans_q_test(data)
        assert isinstance(r, TestResult)
        assert r.statistic == pytest.approx(0.0, abs=1e-10)
        assert r.p_value > 0.99

    def test_different_proportions(self):
        """One column all 1, another all 0 => significant."""
        data = np.column_stack([
            np.ones(30),
            np.zeros(30),
            np.ones(30),
        ])
        r = cochrans_q_test(data)
        assert r.p_value < 0.001

    def test_raises_not_binary(self):
        with pytest.raises(ValueError):
            cochrans_q_test([[1, 2], [3, 4]])

    def test_raises_1d(self):
        with pytest.raises(ValueError):
            cochrans_q_test([1, 0, 1])
