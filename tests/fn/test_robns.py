"""Tests for moirais.fn.robns — Robinson double-residual estimator."""

import numpy as np
import pytest

from moirais.fn.robns import robns


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 300
    Z = rng.standard_normal(n)
    X = rng.standard_normal((n, 1))
    theta_true = 2.0
    g_Z = np.sin(Z)
    Y = X.ravel() * theta_true + g_Z + 0.3 * rng.standard_normal(n)
    return Y, X, Z, theta_true


def test_returns_dict(synth):
    Y, X, Z, _ = synth
    result = robns(Y, X, Z)
    assert isinstance(result, dict)
    for key in ("theta", "se", "ci_lower", "ci_upper", "sigma2", "n", "p", "method"):
        assert key in result


def test_theta_near_true(synth):
    Y, X, Z, theta_true = synth
    result = robns(Y, X, Z)
    assert abs(result["theta"][0] - theta_true) < 1.5


def test_se_positive(synth):
    Y, X, Z, _ = synth
    result = robns(Y, X, Z)
    assert all(s > 0 for s in result["se"])


def test_ci_brackets_theta(synth):
    Y, X, Z, _ = synth
    result = robns(Y, X, Z)
    assert result["ci_lower"][0] <= result["theta"][0] <= result["ci_upper"][0]


def test_epanechnikov_kernel(synth):
    Y, X, Z, _ = synth
    result = robns(Y, X, Z, kernel="epanechnikov")
    assert np.isfinite(result["theta"][0])


def test_method_label(synth):
    Y, X, Z, _ = synth
    result = robns(Y, X, Z)
    assert result["method"] == "Robinson"
