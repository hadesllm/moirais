"""Tests for moirais.fn.pensp -- Penalized spline regression."""

import numpy as np
import pytest

from moirais.fn.pensp import pensp


@pytest.fixture()
def smooth_data():
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 5, 80))
    y = np.sin(x) + rng.standard_normal(80) * 0.3
    return x, y


def test_returns_dict(smooth_data):
    x, y = smooth_data
    result = pensp(x, y)
    assert isinstance(result, dict)
    for k in ("x_grid", "y_hat", "coefficients", "penalty", "n_knots", "gcv_score"):
        assert k in result


def test_y_hat_finite(smooth_data):
    x, y = smooth_data
    result = pensp(x, y)
    assert np.all(np.isfinite(result["y_hat"]))


def test_penalty_positive(smooth_data):
    x, y = smooth_data
    result = pensp(x, y)
    assert result["penalty"] > 0


def test_fixed_penalty(smooth_data):
    x, y = smooth_data
    result = pensp(x, y, penalty=1.0)
    assert result["penalty"] == 1.0


def test_custom_knots(smooth_data):
    x, y = smooth_data
    result = pensp(x, y, n_knots=10)
    assert result["n_knots"] == 10


def test_prediction_grid(smooth_data):
    x, y = smooth_data
    x_new = np.linspace(0, 5, 50)
    result = pensp(x, y, x_new=x_new)
    assert result["y_hat"].shape == (50,)


def test_dimension_error():
    with pytest.raises(ValueError):
        pensp(np.array([1, 2]), np.array([1, 2, 3]))


def test_gcv_score_finite(smooth_data):
    x, y = smooth_data
    result = pensp(x, y)
    assert np.isfinite(result["gcv_score"])


def test_cheatsheet():
    from moirais.fn.pensp import cheatsheet
    assert "spline" in cheatsheet().lower()
