"""Tests for morie.fn.blr -- Bayesian linear regression."""

import numpy as np
import pytest
from morie.fn.blr import bayesian_linear_regression


class TestBayesianLinearRegression:
    def test_recovers_slope(self):
        """With enough data and vague prior, posterior should recover true slope."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 10, n)
        y = 2.0 * x + 1.0 + rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        result = bayesian_linear_regression(X, y, prior_precision=0.001)
        # Slope (index 1) should be near 2.0
        assert abs(result["posterior_mean"][1] - 2.0) < 0.3

    def test_credible_intervals_bracket_true(self):
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(0, 5, n)
        y = 3.0 * x + rng.normal(0, 0.5, n)
        X = np.column_stack([np.ones(n), x])
        result = bayesian_linear_regression(X, y, prior_precision=0.001)
        ci = result["credible_intervals"][1]  # CI for slope
        assert ci[0] < 3.0 < ci[1]

    def test_sigma2_positive(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 2))
        y = X @ [1.5, -0.5] + rng.normal(0, 1, 50)
        result = bayesian_linear_regression(X, y)
        assert result["sigma2_mean"] > 0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows"):
            bayesian_linear_regression(np.ones((10, 2)), np.ones(5))
