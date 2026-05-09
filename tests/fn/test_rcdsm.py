"""Tests for moirais.fn.rcdsm — overall recidivism rate."""

from moirais.fn.rcdsm import recidivism_rate, rcdsm


class TestRecidivismRate:
    def test_returns_dict(self, otis_df):
        result = recidivism_rate(otis_df)
        assert isinstance(result, dict)
        assert "rate" in result
        assert "n_recid" in result
        assert "n_total" in result

    def test_rate_between_0_and_1(self, otis_df):
        result = recidivism_rate(otis_df)
        assert 0.0 <= result["rate"] <= 1.0

    def test_ci_bounds(self, otis_df):
        result = recidivism_rate(otis_df)
        assert result["ci_lower"] <= result["rate"] <= result["ci_upper"]

    def test_alias(self, otis_df):
        assert rcdsm(otis_df) == recidivism_rate(otis_df)
