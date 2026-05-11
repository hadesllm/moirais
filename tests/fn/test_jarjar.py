"""Tests for morie.fn.jarjar -- Jarque-Bera normality test ('How wude!')."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.jarjar import jarque_bera, jarjar
from morie.fn._containers import TestResult


class TestJarJar:
    def test_alias(self):
        assert jarjar is jarque_bera

    def test_normal_data_not_rejected(self):
        """Normal data should not be rejected (p > 0.05)."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        result = jarque_bera(x)
        assert isinstance(result, TestResult)
        assert result.p_value > 0.05
        assert result.test_name == "Jarque-Bera"
        assert result.df == 2

    def test_non_normal_data_rejected(self):
        """Highly skewed data should be rejected (p < 0.05)."""
        rng = np.random.default_rng(42)
        x = rng.exponential(1, 500)
        result = jarque_bera(x)
        assert result.p_value < 0.05

    def test_dataframe_input(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 200)})
        result = jarque_bera(df, col="x")
        assert isinstance(result, TestResult)
        assert result.n == 200

    def test_summary_string(self):
        result = jarque_bera(np.random.default_rng(42).normal(0, 1, 100))
        s = result.summary()
        assert "Jarque-Bera" in s
