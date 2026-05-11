"""Tests for grepl.geron_epsilon_greedy."""
import numpy as np
import pytest
from morie.fn.grepl import geron_epsilon_greedy


def test_grepl_basic():
    """Test basic functionality."""
    Q_s = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_epsilon_greedy(Q_s, eps, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grepl_edge():
    """Test edge cases."""
    Q_s = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_epsilon_greedy(Q_s, eps, seed)
    assert isinstance(result, dict)
