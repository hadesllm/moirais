"""Tests for morie.fn.bcxgm — Box-Cox transformation model via GMM."""

import numpy as np
import pytest

from morie.fn.bcxgm import bcxgm


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 2))
    Y = np.exp(0.5 + X @ np.array([1.0, 0.5]) + 0.2 * rng.standard_normal(n))
    return Y, X


def test_returns_dict(synth):
    Y, X = synth
    result = bcxgm(Y, X)
    assert isinstance(result, dict)
    for key in ("lambda_hat", "beta", "intercept", "se_lambda", "se_beta", "sigma2", "n", "p", "method"):
        assert key in result


def test_lambda_finite(synth):
    Y, X = synth
    result = bcxgm(Y, X)
    assert np.isfinite(result["lambda_hat"])


def test_beta_length(synth):
    Y, X = synth
    result = bcxgm(Y, X)
    assert len(result["beta"]) == 2


def test_positive_sigma2(synth):
    Y, X = synth
    result = bcxgm(Y, X)
    assert result["sigma2"] > 0


def test_negative_y_raises():
    Y = np.array([-1.0, 2.0, 3.0])
    X = np.array([[1], [2], [3]])
    with pytest.raises(ValueError, match="strictly positive"):
        bcxgm(Y, X)


def test_method_label(synth):
    Y, X = synth
    result = bcxgm(Y, X)
    assert result["method"] == "BoxCox_GMM"
