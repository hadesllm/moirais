"""Tests for morie.fn.rskov — risk score overlap."""

from morie.fn.rskov import risk_overlap, rskov


class TestRiskOverlap:
    def test_returns_dict(self, otis_df):
        result = risk_overlap(otis_df)
        assert isinstance(result, dict)
        assert "ks_stat" in result

    def test_ks_stat_between_0_and_1(self, otis_df):
        result = risk_overlap(otis_df)
        assert 0.0 <= result["ks_stat"] <= 1.0

    def test_overlap_coeff_nonneg(self, otis_df):
        result = risk_overlap(otis_df)
        assert result["overlap_coeff"] >= 0.0
