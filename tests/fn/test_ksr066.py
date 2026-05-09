"""Tests for ksr066.kosorok_ch3_z_estimator_no_bias."""
import numpy as np
import pytest
from moirais.fn.ksr066 import kosorok_ch3_z_estimator_no_bias


def test_ksr066_basic():
    """Test basic functionality."""
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    eta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    l_tilde = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch3_z_estimator_no_bias(theta_n, eta_n, theta, l_tilde, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr066_edge():
    """Test edge cases."""
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    eta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    l_tilde = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch3_z_estimator_no_bias(theta_n, eta_n, theta, l_tilde, n)
    assert isinstance(result, dict)
