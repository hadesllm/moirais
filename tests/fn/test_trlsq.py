"""Tests for morie.fn.trlsq -- Total least squares."""

import numpy as np
from morie.fn.trlsq import total_least_squares, trlsq
from morie.fn._containers import DescriptiveResult


class TestTrlsq:
    def test_alias(self):
        assert trlsq is total_least_squares

    def test_simple(self):
        A = np.array([[1], [2], [3]], dtype=float)
        b = np.array([2, 4, 6], dtype=float)
        r = total_least_squares(A, b)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.extra["x"][0] - 2.0) < 0.1

    def test_noisy(self):
        rng = np.random.default_rng(42)
        x_true = np.array([3.0])
        A = rng.standard_normal((20, 1))
        b = (A @ x_true).ravel() + rng.normal(0, 0.01, 20)
        r = total_least_squares(A, b)
        assert abs(r.extra["x"][0] - 3.0) < 0.5
