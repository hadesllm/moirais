"""Tests for moirais.fn.wls -- Weighted least squares."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.wls import weighted_ls, wls
from moirais.fn._containers import RegressionResult


@pytest.fixture()
def wls_data():
    """Heteroskedastic data with known weights."""
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(0, 1, n)
    # Variance proportional to |x|+1
    sigma = np.abs(x) + 1
    y = 2 + 3 * x + rng.normal(0, 1, n) * sigma
    weight = 1.0 / sigma ** 2
    return pd.DataFrame({"y": y, "x": x, "weight": weight})


class TestWls:
    def test_alias(self):
        assert wls is weighted_ls

    def test_returns_regression_result(self, wls_data):
        result = weighted_ls(wls_data, y="y", x="x", w="weight")
        assert isinstance(result, RegressionResult)
        assert result.method == "WLS"

    def test_r_squared_positive(self, wls_data):
        result = weighted_ls(wls_data, y="y", x="x", w="weight")
        assert result.r_squared > 0

    def test_slope_reasonable(self, wls_data):
        """True slope is 3, WLS should recover it."""
        result = weighted_ls(wls_data, y="y", x="x", w="weight")
        assert abs(result.coefficients["x"] - 3.0) < 1.5

    def test_se_finite(self, wls_data):
        result = weighted_ls(wls_data, y="y", x="x", w="weight")
        for s in result.se.values():
            assert np.isfinite(s)

    def test_n_correct(self, wls_data):
        result = weighted_ls(wls_data, y="y", x="x", w="weight")
        assert result.n == 150

    def test_missing_weight_col_raises(self, wls_data):
        with pytest.raises(ValueError, match="Missing columns"):
            weighted_ls(wls_data, y="y", x="x", w="nonexistent")
