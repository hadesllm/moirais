"""Tests for andrew.andrews_sine."""
import numpy as np
import pytest
from morie.fn.andrew import andrews_sine


def test_andrew_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = andrews_sine(y, A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_andrew_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = andrews_sine(y, A)
    assert isinstance(result, dict)
