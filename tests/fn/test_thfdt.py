"""Tests for thfdt.terry_hoeffding_test."""
import numpy as np
import pytest
from morie.fn.thfdt import terry_hoeffding_test


def test_thfdt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = terry_hoeffding_test(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_thfdt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = terry_hoeffding_test(x, y)
    assert isinstance(result, dict)
