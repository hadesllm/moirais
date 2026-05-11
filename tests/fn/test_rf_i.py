"""Tests for morie.fn.rf_i -- iterative OLS regression imputation."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.rf_i import rf_impute


class TestRFImpute:
    def test_no_nans_after_impute(self, missing_df):
        """Result should have no NaN in numeric columns."""
        num_df = missing_df[["x", "y"]].copy()
        result = rf_impute(num_df, n_iter=3)
        assert not result.isna().any().any()

    def test_observed_values_unchanged(self, missing_df):
        """Observed (non-NaN) values must not be changed."""
        num_df = missing_df[["x", "y"]].copy()
        obs_mask = ~num_df.isna()
        result = rf_impute(num_df, n_iter=3)
        for col in ["x", "y"]:
            np.testing.assert_array_almost_equal(
                result.loc[obs_mask[col], col].values,
                num_df.loc[obs_mask[col], col].values,
            )

    def test_imputed_mean_reasonable(self, rng):
        """Imputed values should have a mean close to the true mean."""
        n = 300
        df = pd.DataFrame({
            "a": rng.normal(10, 1, n),
            "b": rng.standard_normal(n),
        })
        mask = rng.random(n) < 0.15
        df.loc[mask, "a"] = np.nan
        result = rf_impute(df, n_iter=5)
        assert abs(result["a"].mean() - 10) < 0.5
