"""Tests for moirais.fn.ocate — CATE by risk score tercile."""

import pandas as pd
from moirais.fn.ocate import otis_cate_risk


class TestOtisCateRisk:
    def test_returns_dataframe(self, otis_df):
        result = otis_cate_risk(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_terciles(self, otis_df):
        result = otis_cate_risk(otis_df)
        assert len(result) >= 2  # at least 2 terciles with data

    def test_columns(self, otis_df):
        result = otis_cate_risk(otis_df)
        for col in ("tercile", "cate", "se", "pval", "n"):
            assert col in result.columns

    def test_risk_bounds(self, otis_df):
        result = otis_cate_risk(otis_df)
        assert (result["risk_lo"] <= result["risk_hi"]).all()
