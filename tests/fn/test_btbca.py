"""Tests for btbca.boot_bca_ci."""
import numpy as np
import pytest
from moirais.fn.btbca import boot_bca_ci


def test_btbca_basic():
    """Test basic functionality."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_bca_ci(theta_hat, theta_b, x, stat, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btbca_edge():
    """Test edge cases."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_bca_ci(theta_hat, theta_b, x, stat, alpha)
    assert isinstance(result, dict)
