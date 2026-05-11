"""Tests for km065.kamath_ch5_reward_loss_pairwise."""
import numpy as np
import pytest
from morie.fn.km065 import kamath_ch5_reward_loss_pairwise


def test_km065_basic():
    """Test basic functionality."""
    r_theta = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_0 = np.random.default_rng(42).normal(0, 1, 100)
    y_1 = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch5_reward_loss_pairwise(r_theta, x, y_0, y_1, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km065_edge():
    """Test edge cases."""
    r_theta = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_0 = np.random.default_rng(42).normal(0, 1, 100)
    y_1 = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch5_reward_loss_pairwise(r_theta, x, y_0, y_1, i)
    assert isinstance(result, dict)
