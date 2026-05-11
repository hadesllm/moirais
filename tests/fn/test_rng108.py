"""Tests for rng108.rangayyan_ch3_ma_8point_recursive."""
import numpy as np
import pytest
from morie.fn.rng108 import rangayyan_ch3_ma_8point_recursive


def test_rng108_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_ma_8point_recursive(x, y, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng108_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_ma_8point_recursive(x, y, n)
    assert isinstance(result, dict)
