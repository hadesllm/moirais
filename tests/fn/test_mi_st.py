"""Tests for morie.fn.mi_st -- strict invariance."""

from morie.fn.mi_st import mi_strict


class TestMiStrict:

    def test_returns_expected_keys(self, mapq_df):
        result = mi_strict(mapq_df, "gender")
        for key in ("level", "fit", "delta_fit", "passed"):
            assert key in result

    def test_level_is_strict(self, mapq_df):
        result = mi_strict(mapq_df, "gender")
        assert result["level"] == "strict"

    def test_passed_is_bool(self, mapq_df):
        result = mi_strict(mapq_df, "gender")
        assert isinstance(result["passed"], bool)
