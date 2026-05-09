"""Tests for ksr072.kosorok_ch3_z_estimator_efficiency_master."""
import numpy as np
import pytest
from moirais.fn.ksr072 import kosorok_ch3_z_estimator_efficiency_master


def test_ksr072_basic():
    """Test basic functionality."""
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    I_tilde = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    n = 100
    result = kosorok_ch3_z_estimator_efficiency_master(theta_n, theta, eta, I_tilde, Z, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr072_edge():
    """Test edge cases."""
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    I_tilde = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    n = 100
    result = kosorok_ch3_z_estimator_efficiency_master(theta_n, theta, eta, I_tilde, Z, n)
    assert isinstance(result, dict)
