"""Tests for moirais.fn.mood -- Mood's median test."""

import numpy as np
import pytest
from moirais.fn.mood import mood_median_test
from moirais.fn._containers import TestResult


class TestMoodMedian:
    def test_identical_groups(self):
        """Same distribution => non-significant."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        r = mood_median_test(a, b)
        assert isinstance(r, TestResult)
        assert r.p_value > 0.05

    def test_shifted_groups(self):
        """Large location shift => significant."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(5, 1, 100)
        r = mood_median_test(a, b)
        assert r.p_value < 0.001

    def test_three_groups(self):
        """Works with k > 2."""
        r = mood_median_test([1, 2, 3], [4, 5, 6], [7, 8, 9])
        assert r.extra["k"] == 3

    def test_raises_single_group(self):
        with pytest.raises(ValueError):
            mood_median_test([1, 2, 3])
