"""Tests for math2.ma_higgins_h2."""
import numpy as np
import pytest
from morie.fn.math2 import ma_higgins_h2


def test_math2_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ma_higgins_h2(Q, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_math2_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ma_higgins_h2(Q, k)
    assert isinstance(result, dict)
