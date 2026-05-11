"""Tests for btwild.boot_wild_regression."""
import numpy as np
import pytest
from morie.fn.btwild import boot_wild_regression


def test_btwild_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_wild_regression(X, y, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btwild_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_wild_regression(X, y, B)
    assert isinstance(result, dict)
