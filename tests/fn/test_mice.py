"""Tests for moirais.fn.mice -- MICE imputation."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.mice import mice_impute


class TestMICE:
    def test_returns_m_datasets(self, missing_df):
        """Should return exactly m completed DataFrames."""
        num_df = missing_df[["x", "y"]].copy()
        result = mice_impute(num_df, m=3, n_iter=5)
        assert len(result) == 3
        for df in result:
            assert isinstance(df, pd.DataFrame)
            assert not df[["x", "y"]].isna().any().any()

    def test_no_missing_raises(self, rng):
        """Should raise if no missing values."""
        df = pd.DataFrame({"a": rng.standard_normal(50), "b": rng.standard_normal(50)})
        with pytest.raises(ValueError, match="No missing"):
            mice_impute(df)

    def test_imputed_means_close(self, rng):
        """Imputed column means should be close to the true mean."""
        n = 500
        true_mean = 5.0
        df = pd.DataFrame({
            "a": rng.normal(true_mean, 1, n),
            "b": rng.standard_normal(n),
        })
        mask = rng.random(n) < 0.15
        df.loc[mask, "a"] = np.nan
        results = mice_impute(df, m=5, n_iter=10)
        pooled_mean = np.mean([d["a"].mean() for d in results])
        assert abs(pooled_mean - true_mean) < 0.5
