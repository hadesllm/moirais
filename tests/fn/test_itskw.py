"""Tests for moirais.fn.itskw — item skewness and kurtosis."""

import numpy as np
import pandas as pd
from moirais.fn.itskw import item_skew_kurt


class TestItemSkewKurt:
    def test_returns_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_skew_kurt(mapq_df[items])
        assert isinstance(result, pd.DataFrame)
        assert "skewness" in result.columns
        assert "kurtosis" in result.columns
        assert len(result) == len(items)

    def test_symmetric_near_zero_skew(self, rng):
        # Large symmetric data
        data = pd.DataFrame({"a": rng.standard_normal(10000) * 2 + 3})
        result = item_skew_kurt(data)
        assert abs(result["skewness"].iloc[0]) < 0.1

    def test_finite_values(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_skew_kurt(mapq_df[items])
        assert result["skewness"].notna().all()
        assert result["kurtosis"].notna().all()

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 3))
        result = item_skew_kurt(data)
        assert len(result) == 3
