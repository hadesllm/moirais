"""Tests for morie.fn.npivs — Nonparametric IV via sieve estimation."""

import numpy as np
import pytest

from morie.fn.npivs import npivs


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 400
    Z = rng.standard_normal(n)
    X = 0.7 * Z + 0.3 * rng.standard_normal(n)
    Y = 2.0 * X + 0.3 * rng.standard_normal(n)
    return Y, X, Z


def test_returns_dict(synth):
    result = npivs(*synth)
    assert isinstance(result, dict)
    for key in ("coefficients", "x_grid", "h_hat", "n", "method"):
        assert key in result


def test_h_hat_grid(synth):
    result = npivs(*synth)
    assert len(result["x_grid"]) == 100
    assert len(result["h_hat"]) == 100


def test_h_hat_finite(synth):
    result = npivs(*synth)
    assert np.all(np.isfinite(result["h_hat"]))


def test_linear_recovery(synth):
    Y, X, Z = synth
    result = npivs(Y, X, Z, n_basis_x=3, n_basis_z=5)
    slope = np.polyfit(result["x_grid"], result["h_hat"], 1)[0]
    assert abs(slope - 2.0) < 1.5


def test_cosine_basis(synth):
    result = npivs(*synth, basis="cosine")
    assert np.all(np.isfinite(result["h_hat"]))


def test_method_label(synth):
    result = npivs(*synth)
    assert result["method"] == "NPIV_Sieve"
