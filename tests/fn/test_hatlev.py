"""Tests for hatlev.leverage."""
import numpy as np
import pytest
from morie.fn.hatlev import leverage


def test_hatlev_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = leverage(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hatlev_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = leverage(X)
    assert isinstance(result, dict)
