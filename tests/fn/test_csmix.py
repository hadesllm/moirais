"""Tests for moirais.fn.csmix — case-control mixture model."""

import numpy as np
import pytest

from moirais.fn.csmix import csmix


@pytest.fixture()
def cc_data():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, 2))
    from scipy.special import expit
    p = expit(0.5 * X[:, 0] - 0.3 * X[:, 1])
    case = rng.binomial(1, p).astype(float)
    return np.zeros(n), X, case


def test_basic_output(cc_data):
    y, X, case = cc_data
    result = csmix(y, X, case)
    assert "beta" in result
    assert "or" in result
    assert result["n_cases"] > 0


def test_or_positive(cc_data):
    y, X, case = cc_data
    result = csmix(y, X, case)
    assert np.all(result["or"] > 0)


def test_beta_direction(cc_data):
    y, X, case = cc_data
    result = csmix(y, X, case)
    assert result["beta"][0] > 0
