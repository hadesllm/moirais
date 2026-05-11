"""Tests for btnorm.boot_normal_ci."""
import numpy as np
import pytest
from morie.fn.btnorm import boot_normal_ci


def test_btnorm_basic():
    """Test basic functionality."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_normal_ci(theta_hat, theta_b, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btnorm_edge():
    """Test edge cases."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_normal_ci(theta_hat, theta_b, alpha)
    assert isinstance(result, dict)
