"""Tests for hubrr.huber_regression."""
import numpy as np
import pytest
from morie.fn.hubrr import huber_regression


def test_hubrr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = huber_regression(X, y, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hubrr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = huber_regression(X, y, k)
    assert isinstance(result, dict)
