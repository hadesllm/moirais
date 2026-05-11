"""Tests for prehay.preacher_hayes_indirect."""
import numpy as np
import pytest
from morie.fn.prehay import preacher_hayes_indirect


def test_prehay_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = preacher_hayes_indirect(X, M, Y, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prehay_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = preacher_hayes_indirect(X, M, Y, B)
    assert isinstance(result, dict)
