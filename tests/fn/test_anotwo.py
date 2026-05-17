"""It does not matter how slowly you go as long as you do not stop. — Confucius"""

import numpy as np
import pandas as pd
import pytest
from morie.fn.anotwo import anova_twoway, anotwo
from morie.fn._containers import TestResult


@pytest.fixture()
def twoway_data():
    """DataFrame with two factors and an outcome."""
    rng = np.random.default_rng(42)
    n = 120
    a = np.repeat(["low", "mid", "high"], n // 3)
    b = np.tile(["ctrl", "treat"], n // 2)
    # Factor a has an effect, factor b does not
    effect_a = np.where(a == "low", 0, np.where(a == "mid", 2, 4))
    y = effect_a + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "a": a, "b": b})


class TestAnotwo:
    def test_alias(self):
        assert anotwo is anova_twoway

    def test_returns_test_result(self, twoway_data):
        result = anova_twoway(twoway_data, y="y", a="a", b="b")
        assert isinstance(result, TestResult)
        assert result.test_name == "Two-way ANOVA (factor A)"

    def test_factor_a_significant(self, twoway_data):
        """Factor a has a real effect, should be significant."""
        result = anova_twoway(twoway_data, y="y", a="a", b="b")
        assert result.p_value < 0.05
        assert result.statistic > 1

    def test_factor_b_not_significant(self, twoway_data):
        """Factor b has no effect, should not be significant."""
        result = anova_twoway(twoway_data, y="y", a="a", b="b")
        assert result.extra["p_b"] > 0.05

    def test_extra_contains_ss(self, twoway_data):
        result = anova_twoway(twoway_data, y="y", a="a", b="b")
        assert "ss_a" in result.extra
        assert "ss_b" in result.extra
        assert "ss_resid" in result.extra

    def test_missing_column_raises(self, twoway_data):
        with pytest.raises(ValueError, match="Missing columns"):
            anova_twoway(twoway_data, y="y", a="a", b="nonexistent")

    def test_custom_column_names(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "outcome": rng.normal(0, 1, 60),
            "group": np.repeat(["A", "B", "C"], 20),
            "cond": np.tile(["X", "Y"], 30),
        })
        result = anova_twoway(df, y="outcome", a="group", b="cond")
        assert isinstance(result, TestResult)
