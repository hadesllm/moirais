"""Tests for morie.fn.mi_mt -- metric invariance."""

from morie.fn.mi_mt import mi_metric
from morie.fn.mi_cf import mi_configural


class TestMiMetric:

    def test_returns_expected_keys(self, mapq_df):
        result = mi_metric(mapq_df, "gender")
        for key in ("level", "fit", "delta_fit", "passed"):
            assert key in result

    def test_level_is_metric(self, mapq_df):
        result = mi_metric(mapq_df, "gender")
        assert result["level"] == "metric"

    def test_delta_fit_with_configural(self, mapq_df):
        cfg = mi_configural(mapq_df, "gender")
        result = mi_metric(mapq_df, "gender", configural_fit=cfg)
        assert "delta_cfi" in result["delta_fit"]
