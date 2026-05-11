"""Tests for morie.fn.oprop — OTIS proportion test."""

from morie.fn.oprop import otis_proportions


class TestOtisProportions:
    def test_returns_dict(self, otis_df):
        result = otis_proportions(otis_df)
        assert isinstance(result, dict)

    def test_keys(self, otis_df):
        result = otis_proportions(otis_df)
        for key in ("chi2", "p_value", "dof", "group_proportions", "overall_proportion"):
            assert key in result

    def test_p_value_bounded(self, otis_df):
        result = otis_proportions(otis_df)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_overall_bounded(self, otis_df):
        result = otis_proportions(otis_df)
        assert 0.0 <= result["overall_proportion"] <= 1.0

    def test_custom_cols(self, otis_df):
        result = otis_proportions(otis_df, col="D", group_col="gender")
        assert result["dof"] >= 1
