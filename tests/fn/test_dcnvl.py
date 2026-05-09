"""Tests for moirais.fn.dcnvl — Deconvolution density estimation."""

import numpy as np
import pytest

from moirais.fn.dcnvl import dcnvl


@pytest.fixture()
def synth():
    rng = np.random.default_rng(42)
    n = 1000
    X_true = rng.standard_normal(n)
    U = 0.5 * rng.standard_normal(n)
    W = X_true + U
    return W


def test_returns_dict(synth):
    result = dcnvl(synth, error_sd=0.5)
    assert isinstance(result, dict)
    for key in ("x_grid", "density", "bandwidth", "n", "method"):
        assert key in result


def test_density_nonnegative(synth):
    result = dcnvl(synth, error_sd=0.5)
    assert np.all(result["density"] >= 0)


def test_density_integrates_to_one(synth):
    result = dcnvl(synth, error_sd=0.5)
    area = np.trapezoid(result["density"], result["x_grid"])
    assert abs(area - 1.0) < 0.15


def test_grid_length(synth):
    result = dcnvl(synth, error_sd=0.5, grid_points=128)
    assert len(result["x_grid"]) == 128
    assert len(result["density"]) == 128


def test_laplace_error(synth):
    result = dcnvl(synth, error_sd=0.5, error_type="laplace")
    assert np.all(result["density"] >= 0)


def test_peak_near_zero(synth):
    result = dcnvl(synth, error_sd=0.5)
    peak_x = result["x_grid"][np.argmax(result["density"])]
    assert abs(peak_x) < 1.0


def test_method_label(synth):
    result = dcnvl(synth, error_sd=0.5)
    assert result["method"] == "FourierDeconvolution"
