"""Tests for fdadj.frontdoor_adjustment."""
import numpy as np
import pytest
from morie.fn.fdadj import frontdoor_adjustment


def test_fdadj_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = frontdoor_adjustment(X, Y, Z, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fdadj_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = frontdoor_adjustment(X, Y, Z, data)
    assert isinstance(result, dict)
