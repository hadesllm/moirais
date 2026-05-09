"""Tests for pace.pace."""
import numpy as np
import pytest
from moirais.fn.pace import pace


def test_pace_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    argvals = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = pace(Y, argvals, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pace_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    argvals = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = pace(Y, argvals, K)
    assert isinstance(result, dict)
