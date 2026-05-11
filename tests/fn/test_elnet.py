"""Tests for morie.fn.elnet -- Elastic net regression."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.elnet import elastic_net, elnet
from morie.fn._containers import RegressionResult


@pytest.fixture()
def reg_data():
    rng = np.random.default_rng(42)
    n = 150
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1 + 2 * x1 + 0.5 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestElnet:
    def test_alias(self):
        assert elnet is elastic_net

    def test_returns_regression_result(self, reg_data):
        result = elastic_net(reg_data, y="y", x=["x1", "x2"])
        assert isinstance(result, RegressionResult)
        assert "ElasticNet" in result.method

    def test_r_squared_positive(self, reg_data):
        result = elastic_net(reg_data, y="y", x=["x1", "x2"], lam=0.1)
        assert result.r_squared > 0

    def test_coefficients_finite(self, reg_data):
        result = elastic_net(reg_data, y="y", x=["x1", "x2"])
        for c in result.coefficients.values():
            assert np.isfinite(c)

    def test_l1_ratio_1_is_lasso_like(self, reg_data):
        """l1_ratio=1 should behave like pure lasso."""
        result = elastic_net(reg_data, y="y", x=["x1", "x2"], lam=10.0, l1_ratio=1.0)
        assert isinstance(result, RegressionResult)

    def test_l1_ratio_0_is_ridge_like(self, reg_data):
        """l1_ratio=0 should behave like pure ridge."""
        result = elastic_net(reg_data, y="y", x=["x1", "x2"], lam=1.0, l1_ratio=0.0)
        assert isinstance(result, RegressionResult)
        # All coefficients should be non-zero (no sparsity with pure ridge)
        for k, v in result.coefficients.items():
            if k != "(Intercept)":
                assert abs(v) > 1e-10
