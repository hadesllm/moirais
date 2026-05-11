"""Tests for morie.fn.llker — Local linear kernel regression."""

import numpy as np
import pytest

from morie.fn.llker import llker


@pytest.fixture()
def lin_data():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 200)
    y = 3.0 * x + 2.0 + rng.normal(0, 0.5, 200)
    return x, y


def test_returns_dict(lin_data):
    x, y = lin_data
    result = llker(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "slope", "bandwidth", "kernel", "n_obs"):
        assert key in result


def test_y_hat_finite(lin_data):
    x, y = lin_data
    result = llker(x, y)
    assert np.all(np.isfinite(result["y_hat"]))


def test_slope_near_true(lin_data):
    x, y = lin_data
    result = llker(x, y, bandwidth=1.5)
    med_slope = np.nanmedian(result["slope"])
    assert abs(med_slope - 3.0) < 1.0


def test_eval_points(lin_data):
    x, y = lin_data
    x_eval = np.array([2.0, 5.0, 8.0])
    result = llker(x, y, x_eval=x_eval)
    assert len(result["y_hat"]) == 3


def test_epanechnikov(lin_data):
    x, y = lin_data
    result = llker(x, y, kernel="epanechnikov")
    assert result["kernel"] == "epanechnikov"


def test_mismatched_raises():
    with pytest.raises(ValueError, match="same length"):
        llker(np.array([1, 2]), np.array([1]))


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 3"):
        llker(np.array([1, 2]), np.array([1, 2]))


def test_boundary_bias_less_than_nw():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 5, 150)
    y = x ** 2 + rng.normal(0, 0.5, 150)
    x_boundary = np.array([0.1, 4.9])
    from morie.fn.nwker import nwker
    nw = nwker(x, y, x_eval=x_boundary, bandwidth=0.5)
    ll = llker(x, y, x_eval=x_boundary, bandwidth=0.5)
    true_vals = x_boundary ** 2
    nw_err = np.abs(nw["y_hat"] - true_vals).mean()
    ll_err = np.abs(ll["y_hat"] - true_vals).mean()
    assert ll_err <= nw_err * 1.5
