"""Tests for moirais.fn.pmm -- predictive mean matching imputation."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.pmm import pmm_impute


class TestPMMImpute:
    def test_no_nans_after_impute(self, missing_df):
        """Result should have no NaN in numeric columns."""
        num_df = missing_df[["x", "y"]].copy()
        result = pmm_impute(num_df, k=3)
        assert not result.isna().any().any()

    def test_imputed_values_from_observed(self, rng):
        """PMM should only use observed values -- no out-of-range imputations."""
        n = 200
        df = pd.DataFrame({
            "a": rng.uniform(0, 10, n),
            "b": rng.standard_normal(n),
        })
        mask = rng.random(n) < 0.15
        df.loc[mask, "a"] = np.nan
        observed_a = df.loc[~mask, "a"].values
        result = pmm_impute(df, k=5)
        imputed_vals = result.loc[mask, "a"].values
        # Every imputed value must be an actually observed value
        for v in imputed_vals:
            assert v in observed_a

    def test_reproducible(self, missing_df):
        """Same seed should give identical results."""
        num_df = missing_df[["x", "y"]].copy()
        r1 = pmm_impute(num_df, seed=99)
        r2 = pmm_impute(num_df, seed=99)
        pd.testing.assert_frame_equal(r1, r2)
