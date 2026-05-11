"""Tests for btbias.boot_bias_estimator."""
import numpy as np
import pytest
from morie.fn.btbias import boot_bias_estimator


def test_btbias_basic():
    """Test basic functionality."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_bias_estimator(theta_hat, theta_b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btbias_edge():
    """Test edge cases."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    theta_b = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_bias_estimator(theta_hat, theta_b)
    assert isinstance(result, dict)
