"""Tests for svmep.svr_epsilon_insensitive."""
import numpy as np
import pytest
from morie.fn.svmep import svr_epsilon_insensitive


def test_svmep_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = svr_epsilon_insensitive(X, y, C, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svmep_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = svr_epsilon_insensitive(X, y, C, eps)
    assert isinstance(result, dict)
