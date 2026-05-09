"""Tests for moirais.fn.irtoc — option characteristic curves."""

import numpy as np
import pandas as pd
from moirais.fn.irtoc import irt_option_curves


class TestIrtOptionCurves:
    def test_returns_dict(self, mapq_df, rng):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        theta = rng.standard_normal(len(mapq_df))
        result = irt_option_curves(mapq_df[items], theta)
        assert isinstance(result, dict)

    def test_single_item(self, mapq_df, rng):
        theta = rng.standard_normal(len(mapq_df))
        result = irt_option_curves(mapq_df, theta, item_col="EE1")
        assert "EE1" in result
        assert isinstance(result["EE1"], pd.DataFrame)
        assert "theta_mid" in result["EE1"].columns

    def test_length_mismatch_raises(self, mapq_df):
        import pytest
        with pytest.raises(ValueError):
            irt_option_curves(mapq_df[["EE1"]], np.array([1, 2, 3]))

    def test_proportions_valid(self, mapq_df, rng):
        theta = rng.standard_normal(len(mapq_df))
        result = irt_option_curves(mapq_df[["EE1"]], theta, n_bins=5)
        df = result["EE1"]
        opt_cols = [c for c in df.columns if c.startswith("option_")]
        # Each row: proportions should sum to ~1
        for _, row in df.iterrows():
            total = sum(row[c] for c in opt_cols)
            assert abs(total - 1.0) < 0.01
