"""Tests for moirais.fn.sefnl — semiparametric efficiency bound."""

import numpy as np
import pytest

from moirais.fn.sefnl import sefnl


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 2))
    T = (X[:, 0] + rng.standard_normal(n) > 0).astype(float)
    Y = 1.0 * T + X[:, 0] + rng.standard_normal(n) * 0.5
    return Y, T, X


def test_bound_positive(synth):
    Y, T, X = synth
    result = sefnl(Y, T, X)
    assert result["efficiency_bound"] > 0


def test_min_se_positive(synth):
    Y, T, X = synth
    result = sefnl(Y, T, X)
    assert result["min_se"] > 0


def test_ci_width_reasonable(synth):
    Y, T, X = synth
    result = sefnl(Y, T, X)
    assert result["min_ci_width"] > 0
    assert result["min_ci_width"] < 10
