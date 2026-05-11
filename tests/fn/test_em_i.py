"""Tests for morie.fn.em_i -- EM algorithm imputation."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.em_i import em_impute


class TestEMImpute:
    def test_no_nans_after_impute(self, missing_df):
        """Result should have no NaN in numeric columns."""
        num_df = missing_df[["x", "y"]].copy()
        result = em_impute(num_df, n_iter=20)
        assert not result[["x", "y"]].isna().any().any()

    def test_convergence_preserves_mean(self, rng):
        """Imputed data mean should be close to the true generating mean."""
        n = 500
        mu = np.array([3.0, 7.0])
        data = rng.multivariate_normal(mu, [[1, 0.5], [0.5, 1]], n)
        df = pd.DataFrame(data, columns=["a", "b"])
        mask = rng.random(n) < 0.12
        df.loc[mask, "a"] = np.nan
        result = em_impute(df, n_iter=50)
        assert abs(result["a"].mean() - 3.0) < 0.3

    def test_too_few_columns_raises(self, rng):
        """Single-column DataFrame should raise ValueError."""
        df = pd.DataFrame({"a": rng.standard_normal(50)})
        df.iloc[0] = np.nan
        with pytest.raises(ValueError, match="at least 2"):
            em_impute(df)
