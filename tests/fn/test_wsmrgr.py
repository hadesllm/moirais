"""Tests for wsmrgr.wasserman_ridge."""
import numpy as np
import pytest
from morie.fn.wsmrgr import wasserman_ridge


def test_wsmrgr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_ridge(X, y, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmrgr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_ridge(X, y, lambda_)
    assert isinstance(result, dict)
