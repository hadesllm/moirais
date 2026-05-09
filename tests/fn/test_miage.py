"""Tests for moirais.fn.miage -- MI by age group."""

from moirais.fn.miage import mi_by_age


class TestMiByAge:

    def test_returns_four_levels(self, mapq_df):
        result = mi_by_age(mapq_df)
        assert len(result) == 4

    def test_configural_has_groups(self, mapq_df):
        result = mi_by_age(mapq_df)
        cfg = result[0]
        assert len(cfg["groups"]) == 4  # 4 age groups in fixture

    def test_levels_ordered(self, mapq_df):
        result = mi_by_age(mapq_df)
        levels = [r["level"] for r in result]
        assert levels == ["configural", "metric", "scalar", "strict"]
