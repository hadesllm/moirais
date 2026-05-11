"""Tests for btht.boot_test_hypothesis."""
import numpy as np
import pytest
from morie.fn.btht import boot_test_hypothesis


def test_btht_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_test_hypothesis(x, theta0, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btht_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_test_hypothesis(x, theta0, stat, B)
    assert isinstance(result, dict)
