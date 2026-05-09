"""Tests for moirais.fn.dmlpl -- DML Partially Linear Model."""

import numpy as np
import pandas as pd
from moirais.fn.dmlpl import dml_plr, dmlpl
from moirais.fn._containers import ESRes


class TestDMLPLR:
    def test_alias(self):
        assert dmlpl is dml_plr

    def test_known_ate_linear(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * t + 0.5 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        result = dml_plr(df, covariates=["x"])
        assert isinstance(result, ESRes)
        assert result.measure == "DML-PLR ATE"
        assert abs(result.estimate - 2.0) < 1.0

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = t + x + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x": x})
        r1 = dml_plr(df, covariates=["x"], seed=77)
        r2 = dml_plr(df, covariates=["x"], seed=77)
        assert r1.estimate == r2.estimate

    def test_multiple_covariates(self):
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 1.5 * t + x1 + 0.5 * x2 + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "x1": x1, "x2": x2})
        result = dml_plr(df, covariates=["x1", "x2"])
        assert result.se > 0
