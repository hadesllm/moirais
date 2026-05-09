"""Tests for ksr062.kosorok_ch3_pathwise_derivative."""
import numpy as np
import pytest
from moirais.fn.ksr062 import kosorok_ch3_pathwise_derivative


def test_ksr062_basic():
    """Test basic functionality."""
    psi = np.random.default_rng(42).normal(0, 1, 100)
    P_t = np.random.default_rng(42).normal(0, 1, 100)
    l_dot = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch3_pathwise_derivative(psi, P_t, l_dot, g, a, theta, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr062_edge():
    """Test edge cases."""
    psi = np.random.default_rng(42).normal(0, 1, 100)
    P_t = np.random.default_rng(42).normal(0, 1, 100)
    l_dot = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch3_pathwise_derivative(psi, P_t, l_dot, g, a, theta, eta)
    assert isinstance(result, dict)
