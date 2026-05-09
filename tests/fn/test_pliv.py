"""Tests for moirais.fn.pliv — Partially Linear IV / 2SLS LATE estimator."""

import math

import numpy as np
import pandas as pd
import pytest

from moirais.fn.pliv import estimate_pliv


@pytest.fixture()
def iv_data():
    """Synthetic IV data with instrument Z, endogenous D, outcome Y."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    # Instrument: correlated with D but not directly with Y
    z = rng.standard_normal(n)
    # Endogenous treatment: depends on Z and X and unobservable u
    u = rng.standard_normal(n)
    d = 0.5 * z + 0.3 * x1 + 0.6 * u + rng.standard_normal(n) * 0.3
    # Outcome: depends on D, X, and u (confounded)
    y = 2.0 * d + 0.5 * x1 + 0.8 * u + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"y": y, "d": d, "z": z, "x1": x1})


def test_returns_dict_with_keys(iv_data):
    """estimate_pliv returns a dict with all required keys."""
    result = estimate_pliv(
        iv_data, treatment="d", outcome="y",
        instrument="z", covariates=["x1"],
    )
    assert isinstance(result, dict)
    for key in ("late", "se", "ci_lower", "ci_upper", "pval", "n_obs", "method"):
        assert key in result, f"Missing key: {key}"


def test_late_is_finite(iv_data):
    """LATE and SE are finite floats."""
    result = estimate_pliv(
        iv_data, treatment="d", outcome="y",
        instrument="z", covariates=["x1"],
    )
    assert math.isfinite(result["late"])
    assert math.isfinite(result["se"])
    assert result["se"] > 0


def test_method_string(iv_data):
    """Method field should be a non-empty string."""
    result = estimate_pliv(
        iv_data, treatment="d", outcome="y",
        instrument="z", covariates=["x1"],
    )
    assert isinstance(result["method"], str)
    assert len(result["method"]) > 0


def test_missing_column_raises():
    """Missing columns should raise ValueError."""
    df = pd.DataFrame({"y": [1], "d": [0], "z": [1]})
    with pytest.raises(ValueError, match="Columns missing"):
        estimate_pliv(
            df, treatment="d", outcome="y",
            instrument="z", covariates=["nonexistent"],
        )


def test_ci_contains_late(iv_data):
    """95% CI should contain the point estimate."""
    result = estimate_pliv(
        iv_data, treatment="d", outcome="y",
        instrument="z", covariates=["x1"],
    )
    assert result["ci_lower"] <= result["late"] <= result["ci_upper"]
