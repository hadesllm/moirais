"""Tests for moirais.fn.odisp — OTIS disparity index."""

from moirais.fn.odisp import otis_disparity_index


class TestOtisDisparityIndex:
    def test_returns_dict(self, otis_df):
        result = otis_disparity_index(otis_df)
        assert isinstance(result, dict)

    def test_keys(self, otis_df):
        result = otis_disparity_index(otis_df)
        for key in ("disparity_index", "max_group", "min_group", "group_means"):
            assert key in result

    def test_index_ge_one(self, otis_df):
        # Use sentence_days (always positive) for a clean test
        result = otis_disparity_index(otis_df, metric_col="sentence_days")
        assert result["disparity_index"] >= 1.0

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"Y": "m", "gender": "g"})
        result = otis_disparity_index(df, metric_col="m", group_col="g")
        assert "group_means" in result
