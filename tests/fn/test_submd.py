"""Tests for morie.fn.submd — substitution estimator."""

import numpy as np
import pytest

from morie.fn.submd import submd


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 2))
    T = (X[:, 0] + rng.standard_normal(n) > 0).astype(float)
    Y = 1.0 * T + X[:, 0] + rng.standard_normal(n) * 0.5
    return Y, T, X


def test_ate_reasonable(synth):
    Y, T, X = synth
    result = submd(Y, T, X)
    assert abs(result["ate"] - 1.0) < 1.0


def test_method_name(synth):
    Y, T, X = synth
    result = submd(Y, T, X)
    assert result["method"] == "Substitution"


def test_se_positive(synth):
    Y, T, X = synth
    result = submd(Y, T, X)
    assert result["se"] >= 0
