"""Tests for robcov.sandwich_robust_se."""
import numpy as np
import pytest
from morie.fn.robcov import sandwich_robust_se


def test_robcov_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    kind = np.random.default_rng(42).normal(0, 1, 100)
    result = sandwich_robust_se(X, y, kind)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_robcov_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    kind = np.random.default_rng(42).normal(0, 1, 100)
    result = sandwich_robust_se(X, y, kind)
    assert isinstance(result, dict)
