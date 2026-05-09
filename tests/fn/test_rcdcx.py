"""Tests for moirais.fn.rcdcx — Cox PH for recidivism."""

from moirais.fn.rcdcx import recidivism_cox, rcdcx


class TestRecidivismCox:
    def test_returns_dict(self, otis_df):
        result = recidivism_cox(otis_df, covariates=["sentence_days"])
        assert isinstance(result, dict)
        assert "coefficients" in result
        assert "hr" in result

    def test_hr_positive(self, otis_df):
        result = recidivism_cox(otis_df, covariates=["sentence_days"])
        for v in result["hr"].values():
            assert v > 0

    def test_n_total(self, otis_df):
        result = recidivism_cox(otis_df, covariates=["sentence_days"])
        assert result["n_total"] == len(otis_df)
