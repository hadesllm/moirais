"""Tests for ksr048.kosorok_ch2_z_master_stochastic_equicontinuity."""
import numpy as np
import pytest
from morie.fn.ksr048 import kosorok_ch2_z_master_stochastic_equicontinuity


def test_ksr048_basic():
    """Test basic functionality."""
    Psi_n = np.random.default_rng(42).normal(0, 1, 100)
    Psi = np.random.default_rng(42).normal(0, 1, 100)
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch2_z_master_stochastic_equicontinuity(Psi_n, Psi, theta_n, theta_0, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr048_edge():
    """Test edge cases."""
    Psi_n = np.random.default_rng(42).normal(0, 1, 100)
    Psi = np.random.default_rng(42).normal(0, 1, 100)
    theta_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_0 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch2_z_master_stochastic_equicontinuity(Psi_n, Psi, theta_n, theta_0, n)
    assert isinstance(result, dict)
