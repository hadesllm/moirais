"""Tests for moirais.fn.exact — exact permutation test."""
import numpy as np
import pytest
from moirais.fn.exact import exact_perm_test


class TestExactPermTest:
    def test_same_distribution(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 40)
        y = rng.normal(0, 1, 40)
        res = exact_perm_test(x, y, n_perm=999)
        assert res.extra["p_value"] > 0.05

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 40)
        y = rng.normal(3, 1, 40)
        res = exact_perm_test(x, y, n_perm=999)
        assert res.extra["p_value"] < 0.05

    def test_observed_statistic(self):
        rng = np.random.default_rng(42)
        x = rng.normal(5, 1, 30)
        y = rng.normal(5, 1, 30)
        res = exact_perm_test(x, y, n_perm=999)
        assert isinstance(res.extra["observed_stat"], float)
