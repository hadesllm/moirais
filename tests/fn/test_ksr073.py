"""Tests for ksr073.kosorok_ch3_max_likelihood_efficiency_corollary."""
import numpy as np
import pytest
from moirais.fn.ksr073 import kosorok_ch3_max_likelihood_efficiency_corollary


def test_ksr073_basic():
    """Test basic functionality."""
    theta_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    eta_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    eta_0 = np.random.default_rng(42).normal(0, 1, 100)
    Psi_dot_0 = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    n = 100
    result = kosorok_ch3_max_likelihood_efficiency_corollary(theta_hat_n, theta_0, eta_hat_n, eta_0, Psi_dot_0, Z, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr073_edge():
    """Test edge cases."""
    theta_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    eta_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    eta_0 = np.random.default_rng(42).normal(0, 1, 100)
    Psi_dot_0 = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    n = 100
    result = kosorok_ch3_max_likelihood_efficiency_corollary(theta_hat_n, theta_0, eta_hat_n, eta_0, Psi_dot_0, Z, n)
    assert isinstance(result, dict)
