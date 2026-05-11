"""Tests for morie.fn.lasso -- Lasso regression (L1)."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.lasso import lasso_regression, lasso
from morie.fn._containers import RegressionResult


@pytest.fixture()
def sparse_data():
    """Data where x1 matters, x2 and x3 are noise."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 2 + 3 * x1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


class TestLasso:
    def test_alias(self):
        assert lasso is lasso_regression

    def test_returns_regression_result(self, sparse_data):
        result = lasso_regression(sparse_data, y="y", x=["x1", "x2", "x3"])
        assert isinstance(result, RegressionResult)
        assert "Lasso" in result.method

    def test_r_squared_positive(self, sparse_data):
        result = lasso_regression(sparse_data, y="y", x=["x1", "x2", "x3"], lam=0.1)
        assert result.r_squared > 0

    def test_coefficients_finite(self, sparse_data):
        result = lasso_regression(sparse_data, y="y", x=["x1", "x2", "x3"], lam=0.1)
        for c in result.coefficients.values():
            assert np.isfinite(c)

    def test_sparsity_with_high_lambda(self, sparse_data):
        """With high lambda, some coefficients should be zeroed out."""
        result = lasso_regression(sparse_data, y="y", x=["x1", "x2", "x3"], lam=50.0)
        zero_count = sum(1 for k, v in result.coefficients.items()
                         if k != "(Intercept)" and abs(v) < 1e-10)
        assert zero_count >= 1  # at least one noise variable zeroed

    def test_n_nonzero_in_extra(self, sparse_data):
        result = lasso_regression(sparse_data, y="y", x=["x1", "x2", "x3"], lam=0.1)
        assert "n_nonzero" in result.extra
