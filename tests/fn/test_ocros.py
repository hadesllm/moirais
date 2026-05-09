"""Tests for moirais.fn.ocros — OTIS cross-tabulation."""

import pandas as pd
from moirais.fn.ocros import otis_crosstab


class TestOtisCrosstab:
    def test_returns_dict(self, otis_df):
        result = otis_crosstab(otis_df)
        assert isinstance(result, dict)

    def test_keys(self, otis_df):
        result = otis_crosstab(otis_df)
        for key in ("table", "chi2", "p_value", "dof", "expected"):
            assert key in result

    def test_table_is_dataframe(self, otis_df):
        result = otis_crosstab(otis_df)
        assert isinstance(result["table"], pd.DataFrame)

    def test_p_value_bounded(self, otis_df):
        result = otis_crosstab(otis_df)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_custom_cols(self, otis_df):
        result = otis_crosstab(otis_df, row_col="gender", col_col="facility_type")
        assert result["dof"] >= 1
