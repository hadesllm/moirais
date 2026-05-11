"""Tests for morie.fn.ctrfn — Control function approach."""

import numpy as np
import pytest

from morie.fn.ctrfn import ctrfn


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 500
    Z = rng.standard_normal(n)
    v = rng.standard_normal(n)
    X = 0.8 * Z + v
    Y = 2.0 * X + 0.5 * v + 0.3 * rng.standard_normal(n)
    return Y, X, Z


def test_returns_dict(synth):
    Y, X, Z = synth
    result = ctrfn(Y, X, Z)
    assert isinstance(result, dict)
    for key in ("beta", "rho", "se_beta", "se_rho", "ci_beta", "hausman_t", "hausman_p", "n", "method"):
        assert key in result


def test_beta_near_true(synth):
    Y, X, Z = synth
    result = ctrfn(Y, X, Z)
    assert abs(result["beta"] - 2.0) < 1.0


def test_rho_nonzero(synth):
    Y, X, Z = synth
    result = ctrfn(Y, X, Z)
    assert abs(result["rho"]) > 0.01


def test_hausman_rejects(synth):
    Y, X, Z = synth
    result = ctrfn(Y, X, Z)
    assert result["hausman_p"] < 0.05


def test_se_positive(synth):
    Y, X, Z = synth
    result = ctrfn(Y, X, Z)
    assert result["se_beta"] > 0
    assert result["se_rho"] > 0


def test_with_exogenous(synth):
    Y, X, Z = synth
    rng = np.random.default_rng(7)
    W = rng.standard_normal((len(Y), 2))
    result = ctrfn(Y, X, Z, W)
    assert np.isfinite(result["beta"])


def test_method_label(synth):
    Y, X, Z = synth
    result = ctrfn(Y, X, Z)
    assert result["method"] == "ControlFunction"
