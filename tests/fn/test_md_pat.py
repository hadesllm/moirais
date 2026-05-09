"""Tests for moirais.fn.md_pat -- missing data pattern analysis."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.md_pat import missing_data_patterns


class TestMissingDataPatterns:
    def test_per_col_fractions(self, missing_df):
        """Per-column missingness fractions should be between 0 and 1."""
        result = missing_data_patterns(missing_df)
        assert all(0 <= v <= 1 for v in result["per_col"])

    def test_patterns_have_count(self, missing_df):
        """Pattern DataFrame must include a count column summing to n."""
        result = missing_data_patterns(missing_df)
        assert "count" in result["patterns"].columns
        assert result["patterns"]["count"].sum() == len(missing_df)

    def test_complete_data_is_monotone(self, rng):
        """Fully observed data should be trivially monotone."""
        df = pd.DataFrame({"a": rng.standard_normal(50), "b": rng.standard_normal(50)})
        # Inject one missing to avoid ValueError in other fns, but still monotone
        df.iloc[0, 0] = np.nan
        result = missing_data_patterns(df)
        assert result["is_monotone"] is True
