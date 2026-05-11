"""Tests for sumP.sum_pool."""
import numpy as np
import pytest
from morie.fn.sumP import sum_pool


def test_sumP_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mode = 'auto'
    result = sum_pool(X, mode)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sumP_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mode = 'auto'
    result = sum_pool(X, mode)
    assert isinstance(result, dict)
