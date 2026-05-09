"""Tests for moirais.fn.dr_ -- Doubly-robust ATE estimator."""

import numpy as np
import pytest
from moirais.fn.dr_ import doubly_robust_ate


class TestDoublyRobust:
    def test_ate_finite(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 2))
        ps = 1 / (1 + np.exp(-X[:, 0]))
        T = rng.binomial(1, ps)
        Y = 0.5 * T + X[:, 0] + rng.normal(0, 1, n)
        result = doubly_robust_ate(Y, T, X)
        assert np.isfinite(result["ate"])
        assert np.isfinite(result["se"])

    def test_ci_brackets_ate(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 1))
        T = rng.binomial(1, 0.5, n)
        Y = 1.0 * T + rng.normal(0, 1, n)
        result = doubly_robust_ate(Y, T, X)
        assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]

    def test_non_binary_treatment_raises(self):
        with pytest.raises(ValueError, match="binary"):
            doubly_robust_ate(np.ones(10), np.array([0, 1, 2]*3 + [0]), np.ones((10, 1)))
