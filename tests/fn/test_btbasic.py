"""Tests for btbasic.boot_basic_ci."""
import numpy as np
import pytest
from morie.fn.btbasic import boot_basic_ci


def test_btbasic_basic():
    """Test basic functionality."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_basic_ci(theta_hat, theta_b, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btbasic_edge():
    """Test edge cases."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = boot_basic_ci(theta_hat, theta_b, alpha)
    assert isinstance(result, dict)
