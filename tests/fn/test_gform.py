"""Tests for morie.fn.gform — G-formula."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.gform import g_formula


class TestGFormula:
    def test_positive_effect(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        a = (rng.uniform(size=n) < 0.5).astype(int)
        y = 0.5 * a + 0.3 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"Y": y, "A": a, "x": x})
        res = g_formula(df, outcome="Y", treatment="A")
        assert abs(res.estimate - 0.5) < 0.3

    def test_returns_ci(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "Y": rng.normal(size=n),
            "A": rng.choice([0, 1], n),
            "x": rng.normal(size=n),
        })
        res = g_formula(df, outcome="Y", treatment="A")
        assert res.ci_lower <= res.ci_upper

    def test_missing_col(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError):
            g_formula(df, outcome="Y", treatment="A")
