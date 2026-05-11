"""Tests for hmavp.geron_average_pool."""
import numpy as np
import pytest
from morie.fn.hmavp import geron_average_pool


def test_hmavp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_average_pool(x, window, stride)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmavp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_average_pool(x, window, stride)
    assert isinstance(result, dict)
