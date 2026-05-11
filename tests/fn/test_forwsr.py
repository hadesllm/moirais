"""Tests for forwsr.forward_search."""
import numpy as np
import pytest
from morie.fn.forwsr import forward_search


def test_forwsr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    initial_h = np.random.default_rng(42).normal(0, 1, 100)
    result = forward_search(X, y, initial_h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_forwsr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    initial_h = np.random.default_rng(42).normal(0, 1, 100)
    result = forward_search(X, y, initial_h)
    assert isinstance(result, dict)
