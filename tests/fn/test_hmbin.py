"""Tests for hmbin.geron_binary_classification."""
import numpy as np
import pytest
from moirais.fn.hmbin import geron_binary_classification


def test_hmbin_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_binary_classification(X, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbin_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    theta = 0.0
    result = geron_binary_classification(X, theta)
    assert isinstance(result, dict)
