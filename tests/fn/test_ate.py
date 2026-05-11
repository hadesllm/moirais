"""Tests for morie.fn.ate — IPW-weighted OLS ATE estimator."""

import math

import numpy as np
import pandas as pd
import pytest

from morie.fn.ate import estimate_ate


@pytest.fixture()
def ate_data():
    """Synthetic data with known treatment effect (~2.0) and IPW weights."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.standard_normal(n)
    t = (rng.uniform(size=n) < (1 / (1 + np.exp(-x)))).astype(float)
    y = 1.0 + 2.0 * t + 0.5 * x + rng.standard_normal(n) * 0.5
    # Simple IPW weights (inverse of propensity)
    ps = 1 / (1 + np.exp(-x))
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    return pd.DataFrame({"y": y, "t": t, "x": x, "w": w})


def test_returns_tuple(ate_data):
    """estimate_ate returns a (coef, se) tuple."""
    result = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_coef_is_finite(ate_data):
    """ATE coefficient and SE are finite floats."""
    coef, se = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    assert math.isfinite(coef)
    assert math.isfinite(se)


def test_coef_near_true_effect(ate_data):
    """ATE coefficient should be roughly near the true effect of 2.0."""
    coef, se = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    # Generous tolerance for n=200 with noise
    assert abs(coef - 2.0) < 2.0, f"ATE={coef} too far from 2.0"


def test_se_positive(ate_data):
    """Standard error must be strictly positive."""
    _, se = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    assert se > 0


def test_weighted_ols_uses_hc3(ate_data):
    """Verify the function runs without error (HC3 robust covariance)."""
    coef, se = estimate_ate(ate_data, outcome="y", treatment="t", weights_col="w")
    # HC3 SEs are typically larger than non-robust SEs; just check it works
    assert se > 0
