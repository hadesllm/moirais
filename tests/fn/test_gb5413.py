"""Tests for gb5413.gibbons_sign_zeros."""
import numpy as np
import pytest
from morie.fn.gb5413 import gibbons_sign_zeros


def test_gb5413_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    median0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_sign_zeros(x, median0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5413_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    median0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_sign_zeros(x, median0)
    assert isinstance(result, dict)
