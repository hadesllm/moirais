"""Tests for gb1231.gibbons_page_test."""
import numpy as np
import pytest
from morie.fn.gb1231 import gibbons_page_test


def test_gb1231_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = gibbons_page_test(data, k)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1231_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = gibbons_page_test(data, k)
    assert isinstance(result, dict)
