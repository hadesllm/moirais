"""Tests for hmeg.geron_epsilon_greedy."""
import numpy as np
import pytest
from morie.fn.hmeg import geron_epsilon_greedy


def test_hmeg_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    epsilon = 1e-6
    result = geron_epsilon_greedy(Q, s, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmeg_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    epsilon = 1e-6
    result = geron_epsilon_greedy(Q, s, epsilon)
    assert isinstance(result, dict)
