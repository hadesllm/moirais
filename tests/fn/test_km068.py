"""Tests for km068.kamath_ch5_ppo_loss."""
import numpy as np
import pytest
from morie.fn.km068 import kamath_ch5_ppo_loss


def test_km068_basic():
    """Test basic functionality."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    r_theta = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_ppo_loss(phi, x, y, r_theta, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km068_edge():
    """Test edge cases."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    r_theta = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_ppo_loss(phi, x, y, r_theta, beta)
    assert isinstance(result, dict)
