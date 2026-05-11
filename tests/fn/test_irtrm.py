"""Tests for morie.fn.irtrm — Rasch model residuals."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.irtrm import irt_rasch_residuals


class TestIrtRaschResiduals:
    def test_returns_dataframe(self, mapq_binary_df, rng):
        k = mapq_binary_df.shape[1]
        params = {col: {"b": float(rng.standard_normal())} for col in mapq_binary_df.columns}
        theta = rng.standard_normal(len(mapq_binary_df))
        result = irt_rasch_residuals(mapq_binary_df, params, theta)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == mapq_binary_df.shape

    def test_standardized_mean_near_zero(self, rng):
        n, k = 500, 10
        # Generate data from Rasch model
        b = rng.standard_normal(k)
        theta = rng.standard_normal(n)
        data = np.zeros((n, k))
        for j in range(k):
            p = 1 / (1 + np.exp(-(theta - b[j])))
            data[:, j] = rng.binomial(1, p)
        params = {f"i{j}": {"b": float(b[j])} for j in range(k)}
        result = irt_rasch_residuals(data, params, theta)
        # Mean standardized residual should be near 0
        assert abs(result.values[~np.isnan(result.values)].mean()) < 0.2

    def test_theta_length_mismatch(self, mapq_binary_df, rng):
        params = {col: {"b": 0.0} for col in mapq_binary_df.columns}
        with pytest.raises(ValueError):
            irt_rasch_residuals(mapq_binary_df, params, np.array([1, 2, 3]))

    def test_item_count_mismatch(self, mapq_binary_df, rng):
        params = {"a": {"b": 0.0}}  # only 1 item
        theta = rng.standard_normal(len(mapq_binary_df))
        with pytest.raises(ValueError):
            irt_rasch_residuals(mapq_binary_df, params, theta)
