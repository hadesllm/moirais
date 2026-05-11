"""Tests for morie.fn.fried -- Friedman test for repeated measures."""

import numpy as np
import pytest
from morie.fn.fried import friedman, fried
from morie.fn._containers import TestResult


class TestFried:
    def test_alias(self):
        assert fried is friedman

    def test_significant_difference(self):
        """Three groups with different locations should be significant."""
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 30)
        g2 = rng.normal(2, 1, 30)
        g3 = rng.normal(4, 1, 30)
        result = friedman(g1, g2, g3)
        assert isinstance(result, TestResult)
        assert result.test_name == "Friedman"
        assert result.p_value < 0.05
        assert result.df == 2

    def test_no_difference(self):
        """Three groups from same distribution -- p should be large."""
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 30)
        g2 = rng.normal(0, 1, 30)
        g3 = rng.normal(0, 1, 30)
        result = friedman(g1, g2, g3)
        assert result.p_value > 0.01  # not necessarily > 0.05 due to randomness

    def test_fewer_than_3_groups_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            friedman([1, 2, 3], [4, 5, 6])

    def test_n_correct(self):
        rng = np.random.default_rng(42)
        g = [rng.normal(0, 1, 20) for _ in range(4)]
        result = friedman(*g)
        assert result.n == 20
        assert result.df == 3
