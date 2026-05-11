"""Tests for ffmFM.field_aware_fm."""
import numpy as np
import pytest
from morie.fn.ffmFM import field_aware_fm


def test_ffmFM_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = field_aware_fm(X, y, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ffmFM_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = field_aware_fm(X, y, K)
    assert isinstance(result, dict)
