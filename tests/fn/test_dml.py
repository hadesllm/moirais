"""Tests for morie.fn.dml — Double Machine Learning (Partially Linear Regression)."""

import numpy as np
import pandas as pd
import pytest

try:
    import doubleml  # noqa: F401
    _HAS_DOUBLEML = True
except ImportError:
    _HAS_DOUBLEML = False

from morie.fn.dml import estimate_double_ml

pytestmark = pytest.mark.skipif(not _HAS_DOUBLEML, reason="DoubleML not installed")


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.3 * x1 - 0.2 * x2)))
    t = rng.binomial(1, prob)
    y = 0.5 * t + 0.3 * x1 + 0.1 * x2 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t, "outcome": y})


def test_returns_fitted_object(synth_data):
    result = estimate_double_ml(
        synth_data, outcome="outcome", treatment="treatment", covariates=["x1", "x2"]
    )
    assert hasattr(result, "coef")
    assert hasattr(result, "se")


def test_coef_is_finite(synth_data):
    result = estimate_double_ml(
        synth_data, outcome="outcome", treatment="treatment", covariates=["x1", "x2"]
    )
    assert np.isfinite(result.coef[0])


def test_se_is_positive(synth_data):
    result = estimate_double_ml(
        synth_data, outcome="outcome", treatment="treatment", covariates=["x1", "x2"]
    )
    assert result.se[0] > 0


def test_custom_random_state(synth_data):
    r1 = estimate_double_ml(
        synth_data, outcome="outcome", treatment="treatment",
        covariates=["x1", "x2"], random_state=123
    )
    r2 = estimate_double_ml(
        synth_data, outcome="outcome", treatment="treatment",
        covariates=["x1", "x2"], random_state=123
    )
    assert r1.coef[0] == pytest.approx(r2.coef[0])
