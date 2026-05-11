"""Tests for morie.fn.sntdp — sentence disparity."""

from morie.fn.sntdp import sentence_disparity, sntdp


class TestSentenceDisparity:
    def test_returns_dict(self, otis_df):
        result = sentence_disparity(otis_df)
        assert isinstance(result, dict)
        assert "max_ratio" in result

    def test_ratio_ge_1(self, otis_df):
        result = sentence_disparity(otis_df)
        assert result["max_ratio"] >= 1.0

    def test_medians_populated(self, otis_df):
        result = sentence_disparity(otis_df)
        assert len(result["medians"]) == otis_df["gender"].nunique()
