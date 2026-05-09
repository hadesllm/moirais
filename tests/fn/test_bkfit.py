"""Tests for moirais.fn.bkfit — Backfitting algorithm for additive models."""

import numpy as np
import pytest

from moirais.fn.bkfit import bkfit


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    Y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * rng.standard_normal(n)
    return Y, X


def test_returns_dict(synth):
    Y, X = synth
    result = bkfit(Y, X)
    assert isinstance(result, dict)
    for key in ("intercept", "components", "fitted", "residuals", "iterations", "converged", "n", "p", "method"):
        assert key in result


def test_components_count(synth):
    Y, X = synth
    result = bkfit(Y, X)
    assert len(result["components"]) == 2


def test_converged(synth):
    Y, X = synth
    result = bkfit(Y, X, max_iter=200)
    assert result["converged"]


def test_fitted_plus_resid(synth):
    Y, X = synth
    result = bkfit(Y, X)
    recon = result["fitted"] + result["residuals"]
    np.testing.assert_allclose(recon, Y, atol=1e-10)


def test_residuals_smaller_than_y(synth):
    Y, X = synth
    result = bkfit(Y, X)
    assert np.std(result["residuals"]) < np.std(Y)


def test_method_label(synth):
    Y, X = synth
    result = bkfit(Y, X)
    assert result["method"] == "Backfitting"
