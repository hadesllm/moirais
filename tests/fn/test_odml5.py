"""Tests for morie.fn.odml5 — OTIS DML volatility."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.odml5 import otis_dml_volatility
from morie.fn._containers import ESRes


class TestOtisDmlVolatility:

    def test_returns_esres(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({"x": rng.standard_normal(n), "volatility": rng.standard_normal(n)})
        df["recidivism"] = (0.3 * df["volatility"] + rng.standard_normal(n) > 0).astype(float)
        result = otis_dml_volatility(df)
        assert isinstance(result, ESRes)

    def test_ci_contains_est(self):
        rng = np.random.default_rng(7)
        n = 200
        df = pd.DataFrame({"x": rng.standard_normal(n), "volatility": rng.standard_normal(n)})
        df["recidivism"] = rng.standard_normal(n)
        result = otis_dml_volatility(df)
        assert result.ci_lower <= result.estimate <= result.ci_upper
