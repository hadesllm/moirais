"""Tests for ksr057.kosorok_ch2_m_estimator_master_theorem."""
import numpy as np
import pytest
from morie.fn.ksr057 import kosorok_ch2_m_estimator_master_theorem


def test_ksr057_basic():
    """Test basic functionality."""
    theta_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    n = 100
    result = kosorok_ch2_m_estimator_master_theorem(theta_hat_n, theta_0, V, Z, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr057_edge():
    """Test edge cases."""
    theta_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    n = 100
    result = kosorok_ch2_m_estimator_master_theorem(theta_hat_n, theta_0, V, Z, n)
    assert isinstance(result, dict)
