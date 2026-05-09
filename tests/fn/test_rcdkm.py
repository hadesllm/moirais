"""Tests for moirais.fn.rcdkm — Kaplan-Meier survival curve."""

from moirais.fn.rcdkm import recidivism_km, rcdkm


class TestRecidivismKM:
    def test_returns_dict(self, otis_df):
        result = recidivism_km(otis_df)
        assert isinstance(result, dict)
        assert "times" in result
        assert "survival" in result

    def test_survival_decreasing(self, otis_df):
        result = recidivism_km(otis_df)
        surv = result["survival"]
        for i in range(1, len(surv)):
            assert surv[i] <= surv[i - 1]

    def test_ci_bounds(self, otis_df):
        result = recidivism_km(otis_df)
        for lo, s, hi in zip(result["ci_lower"], result["survival"], result["ci_upper"]):
            assert lo <= s <= hi
