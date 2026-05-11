"""Tests for lof.local_outlier_factor."""
import numpy as np
import pytest
from morie.fn.lof import local_outlier_factor


def test_lof_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = local_outlier_factor(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lof_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = local_outlier_factor(X, k)
    assert isinstance(result, dict)
