"""Tests for moirais.fn.ochi2 — Chi-squared test of independence."""

from moirais.fn.ochi2 import otis_chi2_test


class TestOtisChi2Test:
    def test_returns_dict(self, otis_df):
        result = otis_chi2_test(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_chi2_test(otis_df)
        for k in ("chi2", "pval", "df", "n", "cramers_v", "contingency_table"):
            assert k in result

    def test_chi2_positive(self, otis_df):
        result = otis_chi2_test(otis_df)
        assert result["chi2"] >= 0

    def test_pval_range(self, otis_df):
        result = otis_chi2_test(otis_df)
        assert 0 <= result["pval"] <= 1

    def test_custom_cols(self, otis_df):
        result = otis_chi2_test(otis_df, row_col="age_group", col_col="gender")
        assert result["n"] > 0
