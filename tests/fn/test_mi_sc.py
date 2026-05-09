"""Tests for moirais.fn.mi_sc -- scalar invariance."""

from moirais.fn.mi_sc import mi_scalar


class TestMiScalar:

    def test_returns_expected_keys(self, mapq_df):
        result = mi_scalar(mapq_df, "gender")
        for key in ("level", "fit", "delta_fit", "passed"):
            assert key in result

    def test_level_is_scalar(self, mapq_df):
        result = mi_scalar(mapq_df, "gender")
        assert result["level"] == "scalar"

    def test_fit_has_cfi(self, mapq_df):
        result = mi_scalar(mapq_df, "gender")
        assert "cfi" in result["fit"]
        assert 0 <= result["fit"]["cfi"] <= 1
