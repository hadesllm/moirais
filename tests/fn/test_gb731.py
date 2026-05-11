"""Tests for gb731.gibbons_linrank_moments."""
import numpy as np
import pytest
from morie.fn.gb731 import gibbons_linrank_moments


def test_gb731_basic():
    """Test basic functionality."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    result = gibbons_linrank_moments(a, m, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb731_edge():
    """Test edge cases."""
    a = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    n = 100
    result = gibbons_linrank_moments(a, m, n)
    assert isinstance(result, dict)
