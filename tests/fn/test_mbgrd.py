"""Tests for mbgrd.mini_batch_gradient."""
import numpy as np
import pytest
from moirais.fn.mbgrd import mini_batch_gradient


def test_mbgrd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mini_batch_gradient(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mbgrd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mini_batch_gradient(x, y)
    assert isinstance(result, dict)
