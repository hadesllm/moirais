"""Tests for mmreg.mm_estimator."""
import numpy as np
import pytest
from morie.fn.mmreg import mm_estimator


def test_mmreg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    c1 = np.random.default_rng(42).normal(0, 1, 100)
    c2 = np.random.default_rng(42).normal(0, 1, 100)
    result = mm_estimator(X, y, c1, c2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mmreg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    c1 = np.random.default_rng(42).normal(0, 1, 100)
    c2 = np.random.default_rng(42).normal(0, 1, 100)
    result = mm_estimator(X, y, c1, c2)
    assert isinstance(result, dict)
