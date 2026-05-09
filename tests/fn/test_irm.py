"""Tests for moirais.fn.irm — Interactive Regression Model via DoubleML."""

import numpy as np
import pandas as pd
import pytest

try:
    import doubleml  # noqa: F401
    _HAS_DOUBLEML = True
except ImportError:
    _HAS_DOUBLEML = False

from moirais.fn.irm import estimate_irm

pytestmark = pytest.mark.skipif(not _HAS_DOUBLEML, reason="DoubleML not installed")


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.3 * x1 - 0.2 * x2)))
    t = rng.binomial(1, prob)
    y = 0.5 * t + 0.3 * x1 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t, "outcome": y})


def test_returns_dict_with_ate(synth_data):
    result = estimate_irm(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert isinstance(result, dict)
    assert "ate" in result


def test_ate_is_finite(synth_data):
    result = estimate_irm(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert np.isfinite(result["ate"])
    assert np.isfinite(result["se"])


def test_has_expected_keys(synth_data):
    result = estimate_irm(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    for key in ("ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert key in result


def test_ci_brackets_ate(synth_data):
    result = estimate_irm(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]
