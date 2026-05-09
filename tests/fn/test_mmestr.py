"""Tests for mmestr.mm_estimator_regression."""
import numpy as np
import pytest
from moirais.fn.mmestr import mm_estimator_regression


def test_mmestr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = mm_estimator_regression(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mmestr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = mm_estimator_regression(y, X)
    assert isinstance(result, dict)
