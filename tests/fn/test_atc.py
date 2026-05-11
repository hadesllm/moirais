"""Tests for morie.fn.atc — Average Treatment Effect on the Controls."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.atc import estimate_atc


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.5 * x1)))
    t = rng.binomial(1, prob)
    y = 0.4 * t + 0.3 * x1 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": t, "outcome": y})


def test_returns_dict(synth_data):
    result = estimate_atc(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert isinstance(result, dict)


def test_atc_is_finite(synth_data):
    result = estimate_atc(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert np.isfinite(result["atc"])
    assert np.isfinite(result["se"])


def test_n_control_matches(synth_data):
    result = estimate_atc(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    expected_n_control = int((synth_data["treatment"] == 0).sum())
    assert result["n_control"] == expected_n_control


def test_has_expected_keys(synth_data):
    result = estimate_atc(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    for key in ("atc", "se", "ci_lower", "ci_upper", "n", "n_treated", "n_control", "method"):
        assert key in result


def test_ci_brackets_atc(synth_data):
    result = estimate_atc(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert result["ci_lower"] <= result["atc"] <= result["ci_upper"]
