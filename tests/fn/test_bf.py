"""Tests for moirais.fn.bf -- Brown-Forsythe test for equality of variances."""

import numpy as np
import pytest
from moirais.fn.bf import brown_forsythe, bf
from moirais.fn._containers import TestResult


class TestBF:
    def test_alias(self):
        assert bf is brown_forsythe

    def test_equal_variances(self):
        """Groups with equal variance should not be rejected."""
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 50)
        g2 = rng.normal(0, 1, 50)
        result = brown_forsythe(g1, g2)
        assert isinstance(result, TestResult)
        assert result.test_name == "Brown-Forsythe"
        assert result.p_value > 0.05

    def test_unequal_variances(self):
        """Groups with very different variance should be rejected."""
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 50)
        g2 = rng.normal(0, 5, 50)
        result = brown_forsythe(g1, g2)
        assert result.p_value < 0.05

    def test_fewer_than_2_groups_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            brown_forsythe([1, 2, 3])

    def test_three_groups(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 30)
        g2 = rng.normal(0, 1, 30)
        g3 = rng.normal(0, 1, 30)
        result = brown_forsythe(g1, g2, g3)
        assert result.df == 2
        assert result.n == 90
