"""Tests for morie.fn.dmlir -- DML Interactive Regression Model."""

import numpy as np
import pandas as pd
from morie.fn.dmlir import dml_irm, dmlir
from morie.fn._containers import ESRes


class TestDMLIRM:
    def test_alias(self):
        assert dmlir is dml_irm

    def test_known_ate(self):
        rng = np.random.default_rng(42)
        n = 600
        x = rng.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-0.5 * x))
        t = rng.binomial(1, ps, n).astype(float)
        y = 1.0 + 2.0 * t + 0.5 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = dml_irm(df, covariates=["x"])
        assert isinstance(result, ESRes)
        assert abs(result.estimate - 2.0) < 1.5

    def test_ci_structure(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 3.0 * t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = dml_irm(df, covariates=["x"])
        assert result.ci_lower < result.estimate < result.ci_upper
        assert result.se > 0
