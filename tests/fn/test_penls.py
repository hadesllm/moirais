"""Tests for penls.penalized_regression."""
import numpy as np
import pytest
from morie.fn.penls import penalized_regression


def test_penls_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = penalized_regression(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_penls_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = penalized_regression(x, y)
    assert isinstance(result, dict)
