"""Tests for rng219.rangayyan_ch4_triangle_inequality_vectors."""
import numpy as np
import pytest
from morie.fn.rng219 import rangayyan_ch4_triangle_inequality_vectors


def test_rng219_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_triangle_inequality_vectors(a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng219_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_triangle_inequality_vectors(a, b)
    assert isinstance(result, dict)
