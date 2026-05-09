"""Tests for grtd0.geron_td_zero_update."""
import numpy as np
import pytest
from moirais.fn.grtd0 import geron_td_zero_update


def test_grtd0_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    next_state = np.random.default_rng(42).normal(0, 1, 100)
    reward = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = geron_td_zero_update(V, state, next_state, reward, alpha, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grtd0_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    state = np.random.default_rng(42).normal(0, 1, 100)
    next_state = np.random.default_rng(42).normal(0, 1, 100)
    reward = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = geron_td_zero_update(V, state, next_state, reward, alpha, gamma)
    assert isinstance(result, dict)
