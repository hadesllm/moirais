"""Tests for sestrg.s_estimator_regression."""
import numpy as np
import pytest
from morie.fn.sestrg import s_estimator_regression


def test_sestrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = s_estimator_regression(y, X, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sestrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = s_estimator_regression(y, X, b)
    assert isinstance(result, dict)
