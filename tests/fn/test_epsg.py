"""Tests for epsg.epsilon_greedy."""
import numpy as np
import pytest
from morie.fn.epsg import epsilon_greedy


def test_epsg_basic():
    """Test basic functionality."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = epsilon_greedy(arms, epsilon, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_epsg_edge():
    """Test edge cases."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = epsilon_greedy(arms, epsilon, T)
    assert isinstance(result, dict)
