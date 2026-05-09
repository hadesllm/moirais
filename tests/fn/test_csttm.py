"""Tests for moirais.fn.csttm — custody time served distribution."""

from moirais.fn.csttm import custody_time_served


class TestCustodyTimeServed:
    def test_returns_dict(self, otis_df):
        result = custody_time_served(otis_df)
        assert isinstance(result, dict)

    def test_keys(self, otis_df):
        result = custody_time_served(otis_df)
        for key in ("mean", "median", "std", "min", "max", "q25", "q75", "n"):
            assert key in result

    def test_min_le_max(self, otis_df):
        result = custody_time_served(otis_df)
        assert result["min"] <= result["max"]

    def test_n_matches_individuals(self, otis_df):
        result = custody_time_served(otis_df)
        assert result["n"] == otis_df["unique_individual_id"].nunique()
