"""Tests for morie.fn.cmprt — compliance rate."""

import pandas as pd
from morie.fn.cmprt import compliance_rate


class TestComplianceRate:
    def test_overall_returns_dict(self, otis_df):
        result = compliance_rate(otis_df)
        assert isinstance(result, dict)
        assert "rate" in result

    def test_rate_bounded(self, otis_df):
        result = compliance_rate(otis_df)
        assert 0.0 <= result["rate"] <= 1.0

    def test_by_group_returns_dataframe(self, otis_df):
        result = compliance_rate(otis_df, group_col="region")
        assert isinstance(result, pd.DataFrame)

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"D": "flag"})
        result = compliance_rate(df, flag_col="flag")
        assert result["n"] == len(df)
