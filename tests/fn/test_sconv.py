"""Tests for morie.fn.sconv -- convergent validity."""

from morie.fn.sconv import subscale_convergent


class TestSubscaleConvergent:

    def test_returns_all_subscales(self, mapq_df):
        result = subscale_convergent(mapq_df)
        assert set(result.keys()) == {"EE", "EA", "UA", "ER"}

    def test_each_has_ave_and_pass(self, mapq_df):
        result = subscale_convergent(mapq_df)
        for name, val in result.items():
            assert "ave" in val
            assert "pass" in val
            assert isinstance(val["pass"], bool)

    def test_ave_nonnegative(self, mapq_df):
        result = subscale_convergent(mapq_df)
        for name, val in result.items():
            assert val["ave"] >= 0
