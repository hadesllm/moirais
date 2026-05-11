"""Tests for morie.fn.rcddl — recidivism DML."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.rcddl import recidivism_dml
from morie.fn._containers import ESRes


class TestRecidivismDml:

    def test_returns_esres(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({"x1": rng.standard_normal(n), "treatment": rng.integers(0, 2, n)})
        df["recidivism"] = (0.3 * df["treatment"] + 0.5 * df["x1"] + rng.standard_normal(n) > 0).astype(int)
        result = recidivism_dml(df)
        assert isinstance(result, ESRes)

    def test_ci_contains_estimate(self):
        rng = np.random.default_rng(7)
        n = 200
        df = pd.DataFrame({"x1": rng.standard_normal(n), "treatment": rng.integers(0, 2, n)})
        df["recidivism"] = (0.5 * df["treatment"] + rng.standard_normal(n) > 0).astype(int)
        result = recidivism_dml(df)
        assert result.ci_lower <= result.estimate <= result.ci_upper
