"""Tests for morie.fn.odml6 — OTIS DML custody."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.odml6 import otis_dml_custody
from morie.fn._containers import ESRes


class TestOtisDmlCustody:

    def test_returns_esres(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({"x": rng.standard_normal(n), "custody_type": rng.integers(0, 2, n)})
        df["recidivism"] = (0.3 * df["custody_type"] + rng.standard_normal(n) > 0).astype(float)
        result = otis_dml_custody(df)
        assert isinstance(result, ESRes)

    def test_ci_contains_est(self):
        rng = np.random.default_rng(7)
        n = 200
        df = pd.DataFrame({"x": rng.standard_normal(n), "custody_type": rng.integers(0, 2, n)})
        df["recidivism"] = rng.standard_normal(n)
        result = otis_dml_custody(df)
        assert result.ci_lower <= result.estimate <= result.ci_upper
