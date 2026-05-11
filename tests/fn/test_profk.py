"""Tests for morie.fn.profk — Profile likelihood."""

import numpy as np
import pytest

from morie.fn.profk import profk


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 * x1 + 0.5 * x2 + rng.standard_normal(n) * 0.5
    X = np.column_stack([x1, x2])
    return y, X


def test_returns_dict(synth):
    result = profk(*synth)
    assert isinstance(result, dict)
    for key in ("theta_hat", "profile_ll", "theta_grid", "ci_lower", "ci_upper", "n", "method"):
        assert key in result


def test_theta_hat_near_truth(synth):
    result = profk(*synth)
    assert abs(result["theta_hat"] - 2.0) < 0.5


def test_ci_contains_truth(synth):
    result = profk(*synth)
    assert result["ci_lower"] <= 2.0 <= result["ci_upper"]


def test_profile_ll_shape(synth):
    result = profk(*synth, n_grid=30)
    assert len(result["profile_ll"]) == 30
    assert len(result["theta_grid"]) == 30


def test_profile_ll_peaked_at_mle(synth):
    result = profk(*synth, n_grid=100)
    peak_idx = np.argmax(result["profile_ll"])
    peak_theta = result["theta_grid"][peak_idx]
    assert abs(peak_theta - result["theta_hat"]) < 0.3


def test_method_label(synth):
    result = profk(*synth)
    assert result["method"] == "ProfileLikelihood"


def test_custom_grid(synth):
    grid = np.linspace(1.0, 3.0, 20)
    result = profk(*synth, theta_grid=grid)
    np.testing.assert_array_equal(result["theta_grid"], grid)
