"""Tests for morie.fn.inspfr — inspection fail rate."""

from morie.fn.inspfr import inspection_fail_rate


class TestInspectionFailRate:
    def test_returns_dict(self, otis_df):
        result = inspection_fail_rate(otis_df)
        assert isinstance(result, dict)

    def test_rate_bounded(self, otis_df):
        result = inspection_fail_rate(otis_df)
        assert 0.0 <= result["fail_rate"] <= 1.0

    def test_threshold_stored(self, otis_df):
        result = inspection_fail_rate(otis_df, threshold=0.5)
        assert result["threshold"] == 0.5

    def test_higher_threshold_more_fails(self, otis_df):
        r1 = inspection_fail_rate(otis_df, threshold=-10.0)
        r2 = inspection_fail_rate(otis_df, threshold=10.0)
        assert r1["n_fail"] <= r2["n_fail"]
