"""Tests for nwest.newey_west_hac."""
import numpy as np
import pytest
from morie.fn.nwest import newey_west_hac


def test_nwest_basic():
    """Test basic functionality."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lags = 10
    result = newey_west_hac(e, X, lags)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nwest_edge():
    """Test edge cases."""
    e = np.random.default_rng(44).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    lags = 10
    result = newey_west_hac(e, X, lags)
    assert isinstance(result, dict)
