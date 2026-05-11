"""Tests for morie.fn.sntln — sentence length distribution."""

from morie.fn.sntln import sentence_length, sntln


class TestSentenceLength:
    def test_returns_dict(self, otis_df):
        result = sentence_length(otis_df)
        assert isinstance(result, dict)
        assert "mean" in result
        assert "median" in result

    def test_min_le_max(self, otis_df):
        result = sentence_length(otis_df)
        assert result["min"] <= result["max"]

    def test_n_matches(self, otis_df):
        result = sentence_length(otis_df)
        assert result["n"] == len(otis_df)
