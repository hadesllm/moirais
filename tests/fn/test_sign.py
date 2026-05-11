"""Tests for morie.fn.sign -- Sign test."""

import numpy as np
import pytest
from morie.fn.sign import sign_test
from morie.fn._containers import TestResult


class TestSign:
    def test_symmetric_differences(self):
        """Equal positive and negative differences => non-significant."""
        x = [1, 2, 3, 4, 5]
        y = [2, 1, 4, 3, 6]  # 2 pos, 2 neg, 1 neg => 2 vs 3
        r = sign_test(x, y)
        assert isinstance(r, TestResult)
        assert r.p_value > 0.05

    def test_all_positive(self):
        """All x > y => significant."""
        x = [10, 20, 30, 40, 50, 60, 70, 80]
        y = [1, 2, 3, 4, 5, 6, 7, 8]
        r = sign_test(x, y)
        assert r.p_value < 0.01

    def test_single_sample(self):
        """Single-sample sign test against median 0."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        r = sign_test(x, mu=0)
        assert r.extra["n_pos"] == 10
        assert r.p_value < 0.01

    def test_all_ties(self):
        """All differences are zero => p=1."""
        r = sign_test([5, 5, 5], [5, 5, 5])
        assert r.p_value == 1.0

    def test_raises_length_mismatch(self):
        with pytest.raises(ValueError):
            sign_test([1, 2], [1, 2, 3])
