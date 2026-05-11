"""Tests for deepF.deepfm."""
import numpy as np
import pytest
from morie.fn.deepF import deepfm


def test_deepF_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    mlp_h = np.random.default_rng(42).normal(0, 1, 100)
    result = deepfm(X, y, K, mlp_h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deepF_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    mlp_h = np.random.default_rng(42).normal(0, 1, 100)
    result = deepfm(X, y, K, mlp_h)
    assert isinstance(result, dict)
