"""Tests for morie.fn.mi_cf -- configural invariance."""

from morie.fn.mi_cf import mi_configural


class TestMiConfigural:

    def test_returns_expected_keys(self, mapq_df):
        result = mi_configural(mapq_df, "gender")
        for key in ("level", "groups", "fit", "group_fits", "passed"):
            assert key in result

    def test_level_is_configural(self, mapq_df):
        result = mi_configural(mapq_df, "gender")
        assert result["level"] == "configural"

    def test_two_groups(self, mapq_df):
        result = mi_configural(mapq_df, "gender")
        assert len(result["groups"]) == 2
