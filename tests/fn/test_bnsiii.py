"""Tests for bnsiii.bound_iii."""
import numpy as np
import pytest
from morie.fn.bnsiii import bound_iii


def test_bnsiii_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_iii(y, X, moments)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnsiii_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_iii(y, X, moments)
    assert isinstance(result, dict)
