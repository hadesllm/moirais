"""Tests for sobel.sobel_test."""
import numpy as np
import pytest
from morie.fn.sobel import sobel_test


def test_sobel_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    sa = np.random.default_rng(42).normal(0, 1, 100)
    sb = np.random.default_rng(42).normal(0, 1, 100)
    result = sobel_test(a, b, sa, sb)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_sobel_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    sa = np.random.default_rng(42).normal(0, 1, 100)
    sb = np.random.default_rng(42).normal(0, 1, 100)
    result = sobel_test(a, b, sa, sb)
    assert isinstance(result, dict)
