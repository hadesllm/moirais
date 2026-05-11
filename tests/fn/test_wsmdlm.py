"""Tests for wsmdlm.wasserman_delta_method."""
import numpy as np
import pytest
from morie.fn.wsmdlm import wasserman_delta_method


def test_wsmdlm_basic():
    """Test basic functionality."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    se = np.random.default_rng(42).normal(0, 1, 100)
    g_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_delta_method(theta_hat, se, g_prime)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmdlm_edge():
    """Test edge cases."""
    theta_hat = np.random.default_rng(42).normal(0, 1, 100)
    se = np.random.default_rng(42).normal(0, 1, 100)
    g_prime = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_delta_method(theta_hat, se, g_prime)
    assert isinstance(result, dict)
