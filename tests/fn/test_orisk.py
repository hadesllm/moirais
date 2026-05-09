"""Tests for moirais.fn.orisk — Univariate risk factor table."""

import pandas as pd
from moirais.fn.orisk import otis_risk_table


class TestOtisRiskTable:
    def test_returns_dataframe(self, otis_df):
        result = otis_risk_table(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, otis_df):
        result = otis_risk_table(otis_df)
        for col in ("covariate", "OR", "ci_lower", "ci_upper", "pval", "n"):
            assert col in result.columns

    def test_has_rows(self, otis_df):
        result = otis_risk_table(otis_df)
        assert len(result) > 0

    def test_custom_covariates(self, otis_df):
        result = otis_risk_table(otis_df, covariates=["alert_mental_health", "sentence_days"])
        assert len(result) == 2
