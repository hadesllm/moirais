"""Tests for morie.fn.mestm — M-estimator with influence function."""

import numpy as np
import pytest

from morie.fn.mestm import mestm, MEstimatorResult


def _huber_rho(xi, theta, k=1.345):
    """Huber loss."""
    r = xi.ravel()[0] - theta[0]
    if abs(r) <= k:
        return 0.5 * r**2
    return k * (abs(r) - 0.5 * k)


def _huber_psi(xi, theta, k=1.345):
    """Huber psi (score)."""
    r = xi.ravel()[0] - theta[0]
    if abs(r) <= k:
        return np.array([r])
    return np.array([k * np.sign(r)])


def _huber_dpsi(xi, theta, k=1.345):
    """Derivative of Huber psi w.r.t. theta."""
    r = xi.ravel()[0] - theta[0]
    if abs(r) <= k:
        return np.array([[-1.0]])
    return np.array([[0.0]])


@pytest.fixture()
def location_data():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((150, 1)) + 5.0
    x[0, 0] = 100.0
    return x


def test_returns_result_type(location_data):
    result = mestm(_huber_rho, _huber_psi, location_data, dpsi=_huber_dpsi,
                   theta0=np.array([4.0]))
    assert isinstance(result, MEstimatorResult)


def test_converges(location_data):
    result = mestm(_huber_rho, _huber_psi, location_data, dpsi=_huber_dpsi,
                   theta0=np.array([4.0]))
    assert result.converged is True


def test_theta_near_true(location_data):
    result = mestm(_huber_rho, _huber_psi, location_data, dpsi=_huber_dpsi,
                   theta0=np.array([4.0]))
    assert abs(result.theta[0] - 5.0) < 1.0


def test_robust_to_outlier(location_data):
    result = mestm(_huber_rho, _huber_psi, location_data, dpsi=_huber_dpsi,
                   theta0=np.array([4.0]))
    mean_val = float(np.mean(location_data))
    assert abs(result.theta[0] - 5.0) < abs(mean_val - 5.0)


def test_se_positive(location_data):
    result = mestm(_huber_rho, _huber_psi, location_data, dpsi=_huber_dpsi,
                   theta0=np.array([4.0]))
    assert result.se[0] > 0


def test_influence_shape(location_data):
    result = mestm(_huber_rho, _huber_psi, location_data, dpsi=_huber_dpsi,
                   theta0=np.array([4.0]))
    assert result.influence_fn.shape == (150, 1)


def test_influence_mean_near_zero(location_data):
    result = mestm(_huber_rho, _huber_psi, location_data, dpsi=_huber_dpsi,
                   theta0=np.array([4.0]))
    assert abs(np.mean(result.influence_fn)) < 0.5


def test_finite_diff_jacobian(location_data):
    result = mestm(_huber_rho, _huber_psi, location_data,
                   theta0=np.array([4.0]))
    assert result.converged is True
    assert abs(result.theta[0] - 5.0) < 1.0


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mestm(_huber_rho, _huber_psi, np.array([]).reshape(0, 1))
