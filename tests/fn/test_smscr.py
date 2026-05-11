"""Tests for morie.fn.smscr — Smoothed maximum score estimator."""

import numpy as np
import pytest

from morie.fn.smscr import smscr


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 400
    beta_true = np.array([1.0, -0.5])
    beta_true /= np.linalg.norm(beta_true)
    X = rng.standard_normal((n, 2))
    idx = X @ beta_true
    Y = (idx + 0.3 * rng.standard_normal(n) > 0).astype(float)
    return Y, X, beta_true


def test_returns_dict(synth):
    Y, X, _ = synth
    result = smscr(Y, X)
    assert isinstance(result, dict)
    for key in ("beta", "score", "bandwidth", "n", "p", "method"):
        assert key in result


def test_unit_norm(synth):
    Y, X, _ = synth
    result = smscr(Y, X)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-6


def test_direction_recovery(synth):
    Y, X, beta_true = synth
    result = smscr(Y, X)
    cosine = abs(np.dot(result["beta"], beta_true))
    assert cosine > 0.5


def test_bandwidth_auto(synth):
    Y, X, _ = synth
    result = smscr(Y, X)
    assert result["bandwidth"] > 0


def test_custom_bandwidth(synth):
    Y, X, _ = synth
    result = smscr(Y, X, bandwidth=0.5)
    assert result["bandwidth"] == 0.5


def test_method_label(synth):
    result = smscr(*synth[:2])
    assert result["method"] == "SmoothedMaximumScore"
