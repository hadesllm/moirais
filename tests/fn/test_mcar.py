"""Tests for morie.fn.mcar -- Little's MCAR test."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.mcar import littles_mcar_test


class TestLittlesMCARTest:
    def test_returns_chi2_df_pvalue(self, missing_df):
        """Result must contain chi2, df, p_value keys."""
        # Use only numeric columns
        result = littles_mcar_test(missing_df, columns=["x", "y"])
        assert "chi2" in result
        assert "df" in result
        assert "p_value" in result
        assert result["chi2"] >= 0
        assert result["df"] >= 0

    def test_mcar_data_high_pvalue(self, rng):
        """Data generated MCAR should usually yield p > 0.01."""
        n = 500
        df = pd.DataFrame({
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        })
        # Inject MCAR missingness
        for col in df.columns:
            mask = rng.random(n) < 0.1
            df.loc[mask, col] = np.nan
        result = littles_mcar_test(df)
        # With truly MCAR data the p-value should usually be high
        assert result["p_value"] > 0.001

    def test_no_missing_raises(self, rng):
        """No missing data should raise ValueError."""
        df = pd.DataFrame({"a": rng.standard_normal(50), "b": rng.standard_normal(50)})
        with pytest.raises(ValueError, match="No missing"):
            littles_mcar_test(df)
