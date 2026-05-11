"""Tests for morie.fn.midif -- delta-fit indices."""

from morie.fn.midif import mi_delta_fit


class TestMiDeltaFit:

    def test_basic(self):
        f1 = {"cfi": 0.96, "rmsea": 0.04, "srmr": 0.05}
        f2 = {"cfi": 0.95, "rmsea": 0.05, "srmr": 0.06}
        result = mi_delta_fit(f1, f2)
        assert abs(result["delta_cfi"] - 0.01) < 1e-10
        assert abs(result["delta_rmsea"] - 0.01) < 1e-10

    def test_passed_true(self):
        f1 = {"cfi": 0.96, "rmsea": 0.04}
        f2 = {"cfi": 0.955, "rmsea": 0.045}
        result = mi_delta_fit(f1, f2)
        assert result["passed"] is True

    def test_passed_false(self):
        f1 = {"cfi": 0.96, "rmsea": 0.04}
        f2 = {"cfi": 0.93, "rmsea": 0.08}
        result = mi_delta_fit(f1, f2)
        assert result["passed"] is False
