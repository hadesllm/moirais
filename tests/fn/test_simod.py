"""Tests for morie.fn.simod — Single-index model estimation."""

import numpy as np
import pytest

from morie.fn.simod import simod


@pytest.fixture()
def sim_data():
    rng = np.random.default_rng(42)
    n = 200
    beta_true = np.array([1.0, 1.0]) / np.sqrt(2)
    X = rng.standard_normal((n, 2))
    z = X @ beta_true
    y = np.sin(z) + rng.normal(0, 0.2, n)
    return y, X, beta_true


def test_returns_dict(sim_data):
    y, X, _ = sim_data
    result = simod(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "g_hat", "index", "converged", "n_iter", "bandwidth", "n_obs"):
        assert key in result


def test_beta_unit_norm(sim_data):
    y, X, _ = sim_data
    result = simod(y, X)
    beta = np.array(result["beta"])
    assert abs(np.linalg.norm(beta) - 1.0) < 1e-6


def test_g_hat_finite(sim_data):
    y, X, _ = sim_data
    result = simod(y, X)
    assert np.all(np.isfinite(result["g_hat"]))


def test_n_obs(sim_data):
    y, X, _ = sim_data
    result = simod(y, X)
    assert result["n_obs"] == 200


def test_converges(sim_data):
    y, X, _ = sim_data
    result = simod(y, X, max_iter=200)
    assert result["n_iter"] > 0


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least"):
        simod(np.ones(3), np.ones((3, 2)))


def test_mismatched_raises():
    with pytest.raises(ValueError, match="!="):
        simod(np.ones(10), np.ones((5, 2)))
