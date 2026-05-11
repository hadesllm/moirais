"""Tests for morie.fn.zestm — Z-estimator (estimating equations)."""

import numpy as np
import pytest

from morie.fn.zestm import zestm, ZEstimatorResult


def _mean_psi(xi, theta):
    """Estimating equation for the mean: psi(x, mu) = x - mu."""
    return xi.ravel() - theta


def _mean_dpsi(xi, theta):
    """Jacobian of mean estimating equation: -I."""
    p = theta.size
    return -np.eye(p)


@pytest.fixture()
def normal_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 1)) + 3.0


def test_returns_result_type(normal_data):
    result = zestm(_mean_psi, _mean_dpsi, normal_data)
    assert isinstance(result, ZEstimatorResult)


def test_converges(normal_data):
    result = zestm(_mean_psi, _mean_dpsi, normal_data)
    assert result.converged is True


def test_theta_near_true_mean(normal_data):
    result = zestm(_mean_psi, _mean_dpsi, normal_data)
    assert abs(result.theta[0] - 3.0) < 0.5


def test_se_positive(normal_data):
    result = zestm(_mean_psi, _mean_dpsi, normal_data)
    assert result.se[0] > 0


def test_se_reasonable(normal_data):
    result = zestm(_mean_psi, _mean_dpsi, normal_data)
    assert result.se[0] < 0.5


def test_cov_shape(normal_data):
    result = zestm(_mean_psi, _mean_dpsi, normal_data)
    assert result.cov.shape == (1, 1)


def test_n_stored(normal_data):
    result = zestm(_mean_psi, _mean_dpsi, normal_data)
    assert result.n == 200


def test_custom_theta0(normal_data):
    result = zestm(_mean_psi, _mean_dpsi, normal_data, theta0=np.array([5.0]))
    assert result.converged is True
    assert abs(result.theta[0] - 3.0) < 0.5


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        zestm(_mean_psi, _mean_dpsi, np.array([]).reshape(0, 1))
