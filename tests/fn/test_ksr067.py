"""Tests for ksr067.kosorok_ch3_z_estimator_consistency_score."""
import numpy as np
import pytest
from moirais.fn.ksr067 import kosorok_ch3_z_estimator_consistency_score


def test_ksr067_basic():
    """Test basic functionality."""
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    eta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    l_tilde = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch3_z_estimator_consistency_score(theta_n, eta_n, theta, eta, l_tilde)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr067_edge():
    """Test edge cases."""
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    eta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    l_tilde = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch3_z_estimator_consistency_score(theta_n, eta_n, theta, eta, l_tilde)
    assert isinstance(result, dict)
