"""Tests for morie.fn.perm -- Permutation test."""

import numpy as np
from morie.fn.perm import permutation_test, perm
from morie.fn._containers import TestResult


class TestPerm:
    def test_alias(self):
        assert perm is permutation_test

    def test_identical_groups_high_p(self):
        """Two samples from same distribution: p should be high."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        result = permutation_test(x, y, n_perm=999)
        assert isinstance(result, TestResult)
        assert result.test_name == "Permutation"
        assert result.p_value > 0.05

    def test_different_groups_low_p(self):
        """Two well-separated groups: p should be very low."""
        rng = np.random.default_rng(42)
        x = rng.normal(10, 1, 50)
        y = rng.normal(0, 1, 50)
        result = permutation_test(x, y, n_perm=999)
        assert result.p_value < 0.01

    def test_alternative_greater(self):
        rng = np.random.default_rng(42)
        x = rng.normal(5, 1, 30)
        y = rng.normal(0, 1, 30)
        result = permutation_test(x, y, n_perm=999, alternative="greater")
        assert result.p_value < 0.05
