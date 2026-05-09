"""Tests for moirais.fn.migen -- MI by gender."""

from moirais.fn.migen import mi_by_gender


class TestMiByGender:

    def test_returns_four_levels(self, mapq_df):
        result = mi_by_gender(mapq_df)
        assert len(result) == 4

    def test_levels_ordered(self, mapq_df):
        result = mi_by_gender(mapq_df)
        levels = [r["level"] for r in result]
        assert levels == ["configural", "metric", "scalar", "strict"]

    def test_each_has_passed(self, mapq_df):
        result = mi_by_gender(mapq_df)
        for r in result:
            assert "passed" in r
            assert isinstance(r["passed"], bool)
