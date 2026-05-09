"""Tests for moirais.fn.kdens — Kernel density estimation."""

import numpy as np
import pytest

from moirais.fn.kdens import kdens


@pytest.fixture()
def normal_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal(500)


def test_returns_dict(normal_data):
    result = kdens(normal_data)
    assert isinstance(result, dict)
    for key in ("x_eval", "density", "bandwidth", "kernel", "n_obs"):
        assert key in result


def test_density_nonnegative(normal_data):
    result = kdens(normal_data)
    assert np.all(result["density"] >= 0)


def test_integrates_to_one(normal_data):
    result = kdens(normal_data, n_grid=512)
    dx = result["x_eval"][1] - result["x_eval"][0]
    integral = np.trapezoid(result["density"], dx=dx)
    assert abs(integral - 1.0) < 0.05


def test_custom_eval_points(normal_data):
    x_eval = np.array([-2, -1, 0, 1, 2], dtype=float)
    result = kdens(normal_data, x_eval=x_eval)
    assert len(result["density"]) == 5


def test_epanechnikov(normal_data):
    result = kdens(normal_data, kernel="epanechnikov")
    assert result["kernel"] == "epanechnikov"
    assert np.all(result["density"] >= 0)


def test_uniform_kernel(normal_data):
    result = kdens(normal_data, kernel="uniform")
    assert np.all(result["density"] >= 0)


def test_peak_near_zero(normal_data):
    result = kdens(normal_data, n_grid=512)
    peak_x = result["x_eval"][np.argmax(result["density"])]
    assert abs(peak_x) < 0.5


def test_unknown_kernel_raises():
    with pytest.raises(ValueError, match="Unknown kernel"):
        kdens(np.array([1, 2, 3]), kernel="cosine")


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 2"):
        kdens(np.array([1.0]))
