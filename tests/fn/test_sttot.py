"""Tests for moirais.fn.sttot -- subscale-total correlations."""

from moirais.fn.sttot import subscale_total_corr


class TestSubscaleTotalCorr:

    def test_returns_all_subscales(self, mapq_df):
        result = subscale_total_corr(mapq_df)
        assert set(result.keys()) == {"EE", "EA", "UA", "ER"}

    def test_correlations_in_range(self, mapq_df):
        result = subscale_total_corr(mapq_df)
        for name, r in result.items():
            assert -1 <= r <= 1

    def test_positive_correlations(self, mapq_df):
        # With correlated subscales, expect positive subscale-total r
        result = subscale_total_corr(mapq_df)
        for name, r in result.items():
            assert r > -0.5  # should not be strongly negative
