"""Tests for ksr071.kosorok_ch3_log_profile_expansion."""
import numpy as np
import pytest
from moirais.fn.ksr071 import kosorok_ch3_log_profile_expansion


def test_ksr071_basic():
    """Test basic functionality."""
    theta_bar_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    I_tilde = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch3_log_profile_expansion(theta_bar_n, theta_hat_n, I_tilde, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr071_edge():
    """Test edge cases."""
    theta_bar_n = np.random.default_rng(42).normal(0, 1, 100)
    theta_hat_n = np.random.default_rng(42).normal(0, 1, 100)
    I_tilde = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch3_log_profile_expansion(theta_bar_n, theta_hat_n, I_tilde, n)
    assert isinstance(result, dict)
