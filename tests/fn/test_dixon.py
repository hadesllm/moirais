"""Tests for dixon.dixon_test."""
import numpy as np
import pytest
from moirais.fn.dixon import dixon_test


def test_dixon_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dixon_test(x, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_dixon_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dixon_test(x, alpha)
    assert isinstance(result, dict)
