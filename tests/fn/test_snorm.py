"""Tests for morie.fn.snorm -- subscale normative tables."""

import pandas as pd
from morie.fn.snorm import subscale_norms


class TestSubscaleNorms:

    def test_returns_dataframe(self, mapq_df):
        result = subscale_norms(mapq_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_has_percentiles(self, mapq_df):
        result = subscale_norms(mapq_df)
        for col in ("p5", "p25", "p50", "p75", "p95"):
            assert col in result.columns

    def test_mean_in_range(self, mapq_df):
        result = subscale_norms(mapq_df)
        assert (result["mean"] >= 1).all()
        assert (result["mean"] <= 5).all()
