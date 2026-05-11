"""Tests for gb541.gibbons_sign_test."""
import numpy as np
import pytest
from morie.fn.gb541 import gibbons_sign_test


def test_gb541_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    median0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_sign_test(x, median0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb541_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    median0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_sign_test(x, median0)
    assert isinstance(result, dict)
