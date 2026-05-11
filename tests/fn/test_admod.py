"""Tests for morie.fn.admod — Additive model via marginal integration."""

import numpy as np
import pytest

from morie.fn.admod import admod


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    Y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * rng.standard_normal(n)
    return Y, X


def test_returns_dict(synth):
    Y, X = synth
    result = admod(Y, X)
    assert isinstance(result, dict)
    for key in ("intercept", "components", "residuals", "n", "p", "method"):
        assert key in result


def test_components_count(synth):
    Y, X = synth
    result = admod(Y, X)
    assert len(result["components"]) == 2


def test_component_grid_size(synth):
    Y, X = synth
    result = admod(Y, X, grid_size=30)
    for comp in result["components"]:
        assert len(comp["x_grid"]) == 30
        assert len(comp["m_hat"]) == 30


def test_residuals_length(synth):
    Y, X = synth
    result = admod(Y, X)
    assert len(result["residuals"]) == 200


def test_residuals_small(synth):
    Y, X = synth
    result = admod(Y, X)
    assert np.std(result["residuals"]) < np.std(Y)


def test_method_label(synth):
    Y, X = synth
    result = admod(Y, X)
    assert result["method"] == "AdditiveModel_MarginalIntegration"
