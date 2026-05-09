"""Tests for moirais.fn.sidxm — Single-index maximum score estimator."""

import numpy as np
import pytest

from moirais.fn.sidxm import sidxm


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 500
    beta_true = np.array([1.0, 0.5])
    beta_true /= np.linalg.norm(beta_true)
    X = rng.standard_normal((n, 2))
    idx = X @ beta_true
    Y = (idx + 0.3 * rng.standard_normal(n) > 0).astype(float)
    return Y, X, beta_true


def test_returns_dict(synth):
    Y, X, _ = synth
    result = sidxm(Y, X)
    assert isinstance(result, dict)
    for key in ("beta", "score", "n", "p", "method"):
        assert key in result


def test_unit_norm(synth):
    Y, X, _ = synth
    result = sidxm(Y, X, n_starts=100)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-6


def test_direction_recovery(synth):
    Y, X, beta_true = synth
    result = sidxm(Y, X, n_starts=100)
    cosine = abs(np.dot(result["beta"], beta_true))
    assert cosine > 0.7


def test_score_positive(synth):
    Y, X, _ = synth
    result = sidxm(Y, X)
    assert result["score"] > 0


def test_n_p_correct(synth):
    Y, X, _ = synth
    result = sidxm(Y, X)
    assert result["n"] == 500
    assert result["p"] == 2


def test_method_label(synth):
    result = sidxm(*synth[:2])
    assert result["method"] == "MaximumScore"
