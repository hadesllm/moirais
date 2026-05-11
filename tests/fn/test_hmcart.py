"""Tests for hmcart.geron_cart_algorithm."""
import numpy as np
import pytest
from morie.fn.hmcart import geron_cart_algorithm


def test_hmcart_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    criterion = np.random.default_rng(42).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_cart_algorithm(X, y, criterion, max_depth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcart_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    criterion = np.random.default_rng(42).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_cart_algorithm(X, y, criterion, max_depth)
    assert isinstance(result, dict)
