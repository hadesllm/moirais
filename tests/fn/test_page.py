"""Tests for moirais.fn.page -- Page's L trend test."""

import numpy as np
import pytest
from moirais.fn.page import page_trend_test
from moirais.fn._containers import TestResult


class TestPage:
    def test_clear_trend(self):
        """Monotone increasing columns => significant L."""
        data = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [1, 3, 5],
            [2, 4, 6],
            [1, 2, 4],
        ])
        r = page_trend_test(data)
        assert isinstance(r, TestResult)
        assert r.p_value < 0.05

    def test_no_trend(self):
        """Random permutations => non-significant."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (20, 3))
        r = page_trend_test(data)
        assert r.p_value > 0.05

    def test_raises_1d(self):
        with pytest.raises(ValueError):
            page_trend_test([1, 2, 3])

    def test_raises_single_col(self):
        with pytest.raises(ValueError):
            page_trend_test([[1], [2], [3]])
