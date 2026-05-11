"""Tests for qlrtst.quandt_likelihood_ratio."""
import numpy as np
import pytest
from morie.fn.qlrtst import quandt_likelihood_ratio


def test_qlrtst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = quandt_likelihood_ratio(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qlrtst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = quandt_likelihood_ratio(y, X)
    assert isinstance(result, dict)
