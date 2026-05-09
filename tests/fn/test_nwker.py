"""Tests for moirais.fn.nwker — Nadaraya-Watson kernel regression."""

import numpy as np
import pytest

from moirais.fn.nwker import nwker


@pytest.fixture()
def sin_data():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 2 * np.pi, 200)
    y = np.sin(x) + rng.normal(0, 0.2, 200)
    return x, y


def test_returns_dict(sin_data):
    x, y = sin_data
    result = nwker(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "bandwidth", "kernel", "n_obs"):
        assert key in result


def test_y_hat_finite(sin_data):
    x, y = sin_data
    result = nwker(x, y)
    assert np.all(np.isfinite(result["y_hat"]))


def test_bandwidth_positive(sin_data):
    x, y = sin_data
    result = nwker(x, y)
    assert result["bandwidth"] > 0


def test_custom_bandwidth(sin_data):
    x, y = sin_data
    result = nwker(x, y, bandwidth=0.5)
    assert result["bandwidth"] == 0.5


def test_eval_points(sin_data):
    x, y = sin_data
    x_eval = np.linspace(0.5, 5.5, 20)
    result = nwker(x, y, x_eval=x_eval)
    assert len(result["y_hat"]) == 20


def test_epanechnikov_kernel(sin_data):
    x, y = sin_data
    result = nwker(x, y, kernel="epanechnikov")
    assert result["kernel"] == "epanechnikov"
    assert np.all(np.isfinite(result["y_hat"]))


def test_uniform_kernel(sin_data):
    x, y = sin_data
    result = nwker(x, y, kernel="uniform")
    assert result["kernel"] == "uniform"


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        nwker(np.array([1, 2]), np.array([1, 2, 3]))


def test_unknown_kernel():
    with pytest.raises(ValueError, match="Unknown kernel"):
        nwker(np.array([1, 2, 3]), np.array([1, 2, 3]), kernel="triangular")


def test_negative_bandwidth():
    with pytest.raises(ValueError, match="positive"):
        nwker(np.array([1, 2, 3]), np.array([1, 2, 3]), bandwidth=-1.0)


def test_recovers_linear():
    rng = np.random.default_rng(99)
    x = rng.uniform(0, 10, 300)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 300)
    result = nwker(x, y, x_eval=np.array([5.0]), bandwidth=1.0)
    assert abs(result["y_hat"][0] - 11.0) < 1.0
