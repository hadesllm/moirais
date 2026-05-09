"""Tests for moirais.fn.sieve -- Sieve estimation."""

import numpy as np
import pytest

from moirais.fn.sieve import sieve


@pytest.fixture()
def regression_data():
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 1, 100))
    y = np.sin(2 * np.pi * x) + rng.standard_normal(100) * 0.2
    return x, y


def test_returns_dict(regression_data):
    x, y = regression_data
    result = sieve(x, y)
    assert isinstance(result, dict)
    for k in ("x_grid", "y_hat", "coefficients", "k", "basis", "residual_var"):
        assert k in result


def test_y_hat_shape(regression_data):
    x, y = regression_data
    result = sieve(x, y)
    assert result["y_hat"].shape == x.shape


def test_residual_var_positive(regression_data):
    x, y = regression_data
    result = sieve(x, y)
    assert result["residual_var"] > 0


def test_cosine_basis(regression_data):
    x, y = regression_data
    result = sieve(x, y, basis="cosine")
    assert result["basis"] == "cosine"
    assert np.all(np.isfinite(result["y_hat"]))


def test_bspline_basis(regression_data):
    x, y = regression_data
    result = sieve(x, y, basis="bspline", k=5)
    assert result["basis"] == "bspline"


def test_custom_k(regression_data):
    x, y = regression_data
    result = sieve(x, y, k=4)
    assert result["k"] == 4


def test_prediction_at_new_points(regression_data):
    x, y = regression_data
    x_new = np.linspace(0, 1, 20)
    result = sieve(x, y, x_new=x_new)
    assert result["y_hat"].shape == (20,)


def test_invalid_basis(regression_data):
    x, y = regression_data
    with pytest.raises(ValueError, match="basis"):
        sieve(x, y, basis="wavelet")


def test_dimension_mismatch():
    with pytest.raises(ValueError):
        sieve(np.array([1, 2]), np.array([1, 2, 3]))


def test_cheatsheet():
    from moirais.fn.sieve import cheatsheet
    assert "sieve" in cheatsheet().lower()
