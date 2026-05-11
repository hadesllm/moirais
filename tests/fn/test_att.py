"""Tests for morie.fn.att — Average Treatment Effect on the Treated."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.att import estimate_att


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
    result = estimate_att(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert isinstance(result, dict)


def test_att_is_finite(synth_data):
    result = estimate_att(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert np.isfinite(result["att"])
    assert np.isfinite(result["se"])


def test_n_treated_matches(synth_data):
    result = estimate_att(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    expected_n_treated = int(synth_data["treatment"].sum())
    assert result["n_treated"] == expected_n_treated


def test_has_expected_keys(synth_data):
    result = estimate_att(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    for key in ("att", "se", "ci_lower", "ci_upper", "n", "n_treated", "n_control", "method"):
        assert key in result


def test_ci_brackets_att(synth_data):
    result = estimate_att(
        synth_data, treatment="treatment", outcome="outcome", covariates=["x1", "x2"]
    )
    assert result["ci_lower"] <= result["att"] <= result["ci_upper"]
