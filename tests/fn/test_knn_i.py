"""Tests for moirais.fn.knn_i -- KNN imputation."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.knn_i import knn_impute


class TestKNNImpute:
    def test_no_nans_after_impute(self, missing_df):
        """Result should have no NaN in numeric columns."""
        num_df = missing_df[["x", "y"]].copy()
        result = knn_impute(num_df, k=3)
        assert not result.isna().any().any()

    def test_observed_values_unchanged(self, missing_df):
        """Observed (non-NaN) values must not be changed."""
        num_df = missing_df[["x", "y"]].copy()
        obs_mask = ~num_df.isna()
        result = knn_impute(num_df, k=3)
        for col in ["x", "y"]:
            np.testing.assert_array_equal(
                result.loc[obs_mask[col], col].values,
                num_df.loc[obs_mask[col], col].values,
            )

    def test_invalid_k_raises(self, missing_df):
        """k < 1 should raise ValueError."""
        num_df = missing_df[["x", "y"]].copy()
        with pytest.raises(ValueError, match="k must be"):
            knn_impute(num_df, k=0)
