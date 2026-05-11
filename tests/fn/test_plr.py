"""Tests for morie.fn.plr — Partially Linear Regression ATE via DoubleML."""

import numpy as np
import pandas as pd
import pytest

try:
    import doubleml  # noqa: F401
    HAS_DML = True
except ImportError:
    HAS_DML = False

pytestmark = pytest.mark.skipif(not HAS_DML, reason="doubleml not installed")

from morie.fn.plr import estimate_plr


@pytest.fixture()
def plr_data():
    """Synthetic PLR data: Y = 1.5*D + g(X) + noise."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    d = 0.3 * x1 + 0.2 * x2 + rng.standard_normal(n) * 0.5
    y = 1.5 * d + 0.8 * x1 - 0.4 * x2 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def test_returns_dict(plr_data):
    """estimate_plr returns a dict with required keys."""
    result = estimate_plr(
        plr_data, treatment="d", outcome="y", covariates=["x1", "x2"]
    )
    assert isinstance(result, dict)
    for key in ("ate", "se", "ci_lower", "ci_upper", "pval", "n_obs"):
        assert key in result, f"Missing key: {key}"


def test_ate_finite(plr_data):
    """ATE and SE should be finite."""
    result = estimate_plr(
        plr_data, treatment="d", outcome="y", covariates=["x1", "x2"]
    )
    assert np.isfinite(result["ate"])
    assert np.isfinite(result["se"])
    assert result["se"] > 0


def test_ci_contains_ate(plr_data):
    """95% CI should contain the point estimate."""
    result = estimate_plr(
        plr_data, treatment="d", outcome="y", covariates=["x1", "x2"]
    )
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_missing_column_raises():
    """Missing columns should raise ValueError."""
    df = pd.DataFrame({"y": [1, 2], "d": [0, 1]})
    with pytest.raises(ValueError, match="Columns missing"):
        estimate_plr(df, treatment="d", outcome="y", covariates=["x_missing"])


def test_n_folds_validation():
    """n_folds < 2 should raise ValueError."""
    df = pd.DataFrame({"y": [1, 2], "d": [0, 1], "x": [0.1, 0.2]})
    with pytest.raises(ValueError, match="n_folds"):
        estimate_plr(df, treatment="d", outcome="y", covariates=["x"], n_folds=1)
