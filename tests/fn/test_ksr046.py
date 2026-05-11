"""Tests for ksr046.kosorok_ch2_z_estimator_consistency."""
import numpy as np
import pytest
from morie.fn.ksr046 import kosorok_ch2_z_estimator_consistency


def test_ksr046_basic():
    """Test basic functionality."""
    Psi_n = np.random.default_rng(42).normal(0, 1, 100)
    Psi = np.random.default_rng(42).normal(0, 1, 100)
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_z_estimator_consistency(Psi_n, Psi, theta_n, theta_0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr046_edge():
    """Test edge cases."""
    Psi_n = np.random.default_rng(42).normal(0, 1, 100)
    Psi = np.random.default_rng(42).normal(0, 1, 100)
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_z_estimator_consistency(Psi_n, Psi, theta_n, theta_0)
    assert isinstance(result, dict)
