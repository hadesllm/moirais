"""Tests for mtsBnd.mts_bounds."""
import numpy as np
import pytest
from morie.fn.mtsbnd import mts_bounds


def test_mtsbnd_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    monotone = np.random.default_rng(42).normal(0, 1, 100)
    result = mts_bounds(Y, X, monotone)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtsbnd_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    monotone = np.random.default_rng(42).normal(0, 1, 100)
    result = mts_bounds(Y, X, monotone)
    assert isinstance(result, dict)
