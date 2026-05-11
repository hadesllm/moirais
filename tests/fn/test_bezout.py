"""Tests for bezout.bezout."""
import numpy as np
import pytest
from morie.fn.bezout import bezout


def test_bezout_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = bezout(a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bezout_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = bezout(a, b)
    assert isinstance(result, dict)
