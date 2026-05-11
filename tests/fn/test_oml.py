"""Tests for morie.fn.oml — DML IRM (ATE/ATT) for OTIS data."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.oml import otdml as oml
from morie.fn._containers import OtDmlR


@pytest.fixture()
def dml_df():
    """Synthetic binary treatment/outcome data with covariates."""
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    d = (x1 + rng.standard_normal(n) * 0.5 > 0).astype(float)
    y = (0.5 * d + 0.3 * x1 + 0.2 * x2 + rng.standard_normal(n) * 0.5 > 0).astype(float)
    return pd.DataFrame({
        "Y": y,
        "D": d,
        "x1": x1,
        "x2": x2,
    })


class TestOtdml:
    """Tests for DML IRM on OTIS data."""

    def test_returns_otdmlr(self, dml_df):
        """Should return OtDmlR dataclass."""
        result = oml(dml_df, outcome="Y", treatment="D", covariates=["x1", "x2"])
        assert isinstance(result, OtDmlR)

    def test_ate_is_finite(self, dml_df):
        """ATE should be a finite number."""
        result = oml(dml_df, outcome="Y", treatment="D", covariates=["x1", "x2"])
        assert np.isfinite(result.ate)

    def test_n_matches_input(self, dml_df):
        """n should match the number of non-missing rows."""
        result = oml(dml_df, outcome="Y", treatment="D", covariates=["x1", "x2"])
        assert result.n == len(dml_df)

    def test_se_positive(self, dml_df):
        """Standard error should be positive."""
        result = oml(dml_df, outcome="Y", treatment="D", covariates=["x1", "x2"])
        assert result.ate_se > 0

    def test_pval_range(self, dml_df):
        """P-value should be between 0 and 1."""
        result = oml(dml_df, outcome="Y", treatment="D", covariates=["x1", "x2"])
        assert 0 <= result.ate_pval <= 1
        assert 0 <= result.att_pval <= 1
