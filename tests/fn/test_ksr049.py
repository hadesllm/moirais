"""Tests for ksr049.kosorok_ch2_z_master_linearization."""
import numpy as np
import pytest
from morie.fn.ksr049 import kosorok_ch2_z_master_linearization


def test_ksr049_basic():
    """Test basic functionality."""
    Psi_dot = np.random.default_rng(42).normal(0, 1, 100)
    Psi_n = np.random.default_rng(42).normal(0, 1, 100)
    Psi = np.random.default_rng(42).normal(0, 1, 100)
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch2_z_master_linearization(Psi_dot, Psi_n, Psi, theta_n, theta_0, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr049_edge():
    """Test edge cases."""
    Psi_dot = np.random.default_rng(42).normal(0, 1, 100)
    Psi_n = np.random.default_rng(42).normal(0, 1, 100)
    Psi = np.random.default_rng(42).normal(0, 1, 100)
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch2_z_master_linearization(Psi_dot, Psi_n, Psi, theta_n, theta_0, n)
    assert isinstance(result, dict)
