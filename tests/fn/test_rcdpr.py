"""Tests for morie.fn.rcdpr — logistic regression for recidivism."""

from morie.fn.rcdpr import recidivism_predictors, rcdpr


class TestRecidivismPredictors:
    def test_returns_dict(self, otis_df):
        result = recidivism_predictors(otis_df, covariates=["sentence_days"])
        assert isinstance(result, dict)
        assert "coefficients" in result
        assert "odds_ratios" in result

    def test_odds_ratios_positive(self, otis_df):
        result = recidivism_predictors(otis_df, covariates=["sentence_days"])
        for v in result["odds_ratios"].values():
            assert v > 0

    def test_n_total(self, otis_df):
        result = recidivism_predictors(otis_df, covariates=["sentence_days"])
        assert result["n_total"] > 0
