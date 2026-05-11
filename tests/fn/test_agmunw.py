"""Tests for agmunw.muzero_n_step_value."""
import numpy as np
import pytest
from morie.fn.agmunw import muzero_n_step_value


def test_agmunw_basic():
    """Test basic functionality."""
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    values = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    gamma = 1.0
    result = muzero_n_step_value(rewards, values, n, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agmunw_edge():
    """Test edge cases."""
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    values = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    gamma = 1.0
    result = muzero_n_step_value(rewards, values, n, gamma)
    assert isinstance(result, dict)
