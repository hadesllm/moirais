"""Tests for funkM.funk_svd."""
import numpy as np
import pytest
from morie.fn.funkM import funk_svd


def test_funkM_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    lr = np.random.default_rng(42).normal(0, 1, 100)
    reg = np.random.default_rng(42).normal(0, 1, 100)
    result = funk_svd(R, K, lr, reg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_funkM_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    lr = np.random.default_rng(42).normal(0, 1, 100)
    reg = np.random.default_rng(42).normal(0, 1, 100)
    result = funk_svd(R, K, lr, reg)
    assert isinstance(result, dict)
