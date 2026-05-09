"""Tests for moirais.fn.arisc -- Adjusted Rand index."""

import numpy as np
from moirais.fn.arisc import adjusted_rand_index, arisc
from moirais.fn._containers import DescriptiveResult


class TestAdjustedRandIndex:
    def test_alias(self):
        assert arisc is adjusted_rand_index

    def test_returns_result(self):
        y = np.array([0, 0, 1, 1])
        yp = np.array([0, 0, 1, 1])
        res = adjusted_rand_index(y, yp)
        assert isinstance(res, DescriptiveResult)

    def test_perfect_agreement(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        res = adjusted_rand_index(y, y)
        assert abs(res.value - 1.0) < 1e-6

    def test_random_near_zero(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 5, 100)
        yp = rng.integers(0, 5, 100)
        res = adjusted_rand_index(y, yp)
        assert abs(res.value) < 0.3
