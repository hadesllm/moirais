"""Tests for moirais.fn.cstgp — custody gender parity index."""

from moirais.fn.cstgp import custody_gender_parity


class TestCustodyGenderParity:
    def test_returns_dict(self, otis_df):
        result = custody_gender_parity(otis_df)
        assert isinstance(result, dict)

    def test_parity_bounded(self, otis_df):
        result = custody_gender_parity(otis_df)
        assert 0.0 <= result["parity_index"] <= 1.0

    def test_group_means_present(self, otis_df):
        result = custody_gender_parity(otis_df)
        assert len(result["group_means"]) == otis_df["gender"].nunique()

    def test_custom_cols(self, otis_df):
        result = custody_gender_parity(otis_df, gender_col="gender", metric_col="Y")
        assert result["n_groups"] >= 2
