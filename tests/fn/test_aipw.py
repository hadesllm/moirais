"""Tests for morie.fn.aipw — Augmented IPW doubly-robust ATE estimator."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.aipw import estimate_aipw


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.4 * x1 - 0.2 * x2)))
    t = rng.binomial(1, prob)
    y = (0.3 * t + 0.2 * x1 + rng.standard_normal(n) * 0.5 > 0.2).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t, "outcome": y})


def test_returns_dict_with_ate(synth_data):
    result = estimate_aipw(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert isinstance(result, dict)
    assert "ate" in result


def test_ate_is_finite(synth_data):
    result = estimate_aipw(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert np.isfinite(result["ate"])
    assert np.isfinite(result["se"])


def test_ci_brackets_ate(synth_data):
    result = estimate_aipw(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_has_expected_keys(synth_data):
    result = estimate_aipw(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    for key in ("ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert key in result


def test_linear_outcome_model(synth_data):
    result = estimate_aipw(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1", "x2"],
        outcome_model="linear",
    )
    assert np.isfinite(result["ate"])
