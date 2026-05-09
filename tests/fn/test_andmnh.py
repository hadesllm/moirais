"""Tests for andmnh.andrews_monahan_hac."""
import numpy as np
import pytest
from moirais.fn.andmnh import andrews_monahan_hac


def test_andmnh_basic():
    """Test basic functionality."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = andrews_monahan_hac(e, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_andmnh_edge():
    """Test edge cases."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = andrews_monahan_hac(e, X)
    assert isinstance(result, dict)
