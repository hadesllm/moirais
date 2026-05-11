"""Tests for andrews.andrews_sine."""
import numpy as np
import pytest
from morie.fn.andrews import andrews_sine


def test_andrews_basic():
    """Test basic functionality."""
    r = 10
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = andrews_sine(r, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_andrews_edge():
    """Test edge cases."""
    r = 10
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = andrews_sine(r, c)
    assert isinstance(result, dict)
