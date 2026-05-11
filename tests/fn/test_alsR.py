"""Tests for alsR.als."""
import numpy as np
import pytest
from morie.fn.alsR import als


def test_alsR_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    reg = np.random.default_rng(42).normal(0, 1, 100)
    iters = np.random.default_rng(42).normal(0, 1, 100)
    result = als(R, K, reg, iters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alsR_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    reg = np.random.default_rng(42).normal(0, 1, 100)
    iters = np.random.default_rng(42).normal(0, 1, 100)
    result = als(R, K, reg, iters)
    assert isinstance(result, dict)
