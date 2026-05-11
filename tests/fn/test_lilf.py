"""Tests for morie.fn.lilf -- Lilliefors test."""

import numpy as np
import pytest
from morie.fn.lilf import lilliefors_test
from morie.fn._containers import TestResult


class TestLilliefors:
    def test_normal_data(self):
        """Data from normal distribution => non-significant."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = lilliefors_test(x)
        assert isinstance(r, TestResult)
        assert r.p_value > 0.05

    def test_non_normal(self):
        """Exponential data => significant departure from normality."""
        rng = np.random.default_rng(42)
        x = rng.exponential(1, 200)
        r = lilliefors_test(x)
        assert r.p_value < 0.05

    def test_zero_variance(self):
        """Constant data => p=1."""
        r = lilliefors_test([5, 5, 5, 5])
        assert r.p_value == 1.0

    def test_raises_small_n(self):
        with pytest.raises(ValueError):
            lilliefors_test([1, 2, 3])
