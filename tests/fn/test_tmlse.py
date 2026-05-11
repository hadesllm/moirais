"""Tests for morie.fn.tmlse — TMLE standard error."""

import numpy as np
import pytest

from morie.fn.tmlse import tmlse


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 2))
    T = (X[:, 0] + rng.standard_normal(n) > 0).astype(float)
    Y = 1.0 * T + X[:, 0] + rng.standard_normal(n) * 0.5
    return Y, T, X


def test_se_positive(synth):
    Y, T, X = synth
    result = tmlse(Y, T, X)
    assert result["se"] > 0


def test_ci_contains_ate(synth):
    Y, T, X = synth
    result = tmlse(Y, T, X)
    assert result["ci_lower"] < result["ate"] < result["ci_upper"]


def test_influence_values_shape(synth):
    Y, T, X = synth
    result = tmlse(Y, T, X)
    assert len(result["influence_values"]) == 500
