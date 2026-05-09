"""Tests for moirais.fn.cfacm -- nested CFA model comparison."""

from moirais.fn.cfacm import cfa_compare


class TestCfaCompare:

    def test_basic_comparison(self):
        fit1 = {"chi2": 100, "df": 50, "cfi": 0.95}
        fit2 = {"chi2": 120, "df": 55, "cfi": 0.93}
        result = cfa_compare(fit1, fit2)
        assert result["delta_chi2"] == 20.0
        assert result["delta_df"] == 5
        assert 0 <= result["p_value"] <= 1

    def test_delta_cfi_sign(self):
        fit1 = {"chi2": 100, "df": 50, "cfi": 0.96}
        fit2 = {"chi2": 110, "df": 55, "cfi": 0.94}
        result = cfa_compare(fit1, fit2)
        assert result["delta_cfi"] > 0  # constrained is worse

    def test_significance_flag(self):
        fit1 = {"chi2": 50, "df": 40, "cfi": 0.98}
        fit2 = {"chi2": 51, "df": 41, "cfi": 0.97}
        result = cfa_compare(fit1, fit2, alpha=0.05)
        assert isinstance(result["significant"], bool)
