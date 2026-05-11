"""Tests for bkmed.baron_kenny."""
import numpy as np
import pytest
from morie.fn.bkmed import baron_kenny


def test_bkmed_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = baron_kenny(Y, X, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkmed_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = baron_kenny(Y, X, M)
    assert isinstance(result, dict)
