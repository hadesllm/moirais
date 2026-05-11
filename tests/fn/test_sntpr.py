"""Tests for morie.fn.sntpr — sentence percentiles."""

from morie.fn.sntpr import sentence_percentiles, sntpr


class TestSentencePercentiles:
    def test_returns_dict(self, otis_df):
        result = sentence_percentiles(otis_df)
        assert isinstance(result, dict)
        assert "p50" in result

    def test_percentiles_ordered(self, otis_df):
        result = sentence_percentiles(otis_df)
        assert result["p5"] <= result["p25"] <= result["p50"]
        assert result["p50"] <= result["p75"] <= result["p95"]

    def test_n_matches(self, otis_df):
        result = sentence_percentiles(otis_df)
        assert result["n"] == len(otis_df)
