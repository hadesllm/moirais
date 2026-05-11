"""Tests for morie.fn.ocomp — OTIS group comparison."""

from morie.fn.ocomp import otis_group_compare


class TestOtisGroupCompare:
    def test_returns_dict(self, otis_df):
        result = otis_group_compare(otis_df)
        assert isinstance(result, dict)

    def test_two_groups_uses_t(self, otis_df):
        result = otis_group_compare(otis_df, group_col="gender")
        assert result["test"] == "welch_t"
        assert result["n_groups"] == 2

    def test_three_groups_uses_anova(self, otis_df):
        result = otis_group_compare(otis_df, group_col="age_group")
        assert result["test"] == "anova_f"
        assert result["n_groups"] == 3

    def test_p_value_bounded(self, otis_df):
        result = otis_group_compare(otis_df)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"Y": "m", "gender": "g"})
        result = otis_group_compare(df, metric_col="m", group_col="g")
        assert "effect_size" in result
