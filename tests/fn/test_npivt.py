"""Tests for moirais.fn.npivt — Nonparametric IV via Tikhonov."""

import numpy as np
import pytest

from moirais.fn.npivt import npivt


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 400
    Z = rng.standard_normal(n)
    X = 0.7 * Z + 0.3 * rng.standard_normal(n)
    Y = np.sin(X) + 0.3 * rng.standard_normal(n)
    return Y, X, Z


def test_returns_dict(synth):
    result = npivt(*synth)
    assert isinstance(result, dict)
    for key in ("x_grid", "h_hat", "alpha_reg", "n", "method"):
        assert key in result


def test_h_hat_length(synth):
    result = npivt(*synth, n_basis=15)
    assert len(result["h_hat"]) == 15
    assert len(result["x_grid"]) == 15


def test_h_hat_finite(synth):
    result = npivt(*synth)
    assert np.all(np.isfinite(result["h_hat"]))


def test_n_correct(synth):
    result = npivt(*synth)
    assert result["n"] == 400


def test_method_label(synth):
    result = npivt(*synth)
    assert result["method"] == "NPIV_Tikhonov"


def test_custom_alpha(synth):
    result = npivt(*synth, alpha_reg=0.1)
    assert result["alpha_reg"] == 0.1
