"""Tests for morie.fn.bal -- Balance diagnostics."""

import numpy as np
import pytest
from morie.fn.bal import balance_diagnostics


class TestBalanceDiagnostics:
    def test_balanced_groups_small_smd(self):
        """Identical group distributions -> SMD near 0."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 2))
        T = rng.binomial(1, 0.5, n)
        results = balance_diagnostics(X, T)
        for r in results:
            assert abs(r["smd_raw"]) < 0.3

    def test_imbalanced_groups_large_smd(self):
        """Groups with different means -> large SMD."""
        X = np.array([[0.1], [0.2], [0.3], [4.8], [5.0], [5.2]])
        T = np.array([0, 0, 0, 1, 1, 1])
        results = balance_diagnostics(X, T)
        assert abs(results[0]["smd_raw"]) > 1.0

    def test_weighted_smd_returned(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 1))
        T = rng.binomial(1, 0.5, n)
        w = np.ones(n)
        results = balance_diagnostics(X, T, weights=w)
        assert results[0]["smd_weighted"] is not None

    def test_custom_names(self):
        X = np.ones((10, 2))
        T = np.array([0]*5 + [1]*5)
        results = balance_diagnostics(X, T, covariate_names=["age", "income"])
        assert results[0]["variable"] == "age"
        assert results[1]["variable"] == "income"
