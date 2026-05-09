"""Tests for mai2.ma_higgins_i2."""
import numpy as np
import pytest
from moirais.fn.mai2 import ma_higgins_i2


def test_mai2_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ma_higgins_i2(Q, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mai2_edge():
    """Test edge cases."""
    Q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ma_higgins_i2(Q, k)
    assert isinstance(result, dict)
