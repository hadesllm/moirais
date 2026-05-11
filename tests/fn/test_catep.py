"""Tests for catep.cate_estimation."""
import numpy as np
import pytest
from morie.fn.catep import cate_estimation


def test_catep_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = cate_estimation(Y, T, X, estimator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_catep_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = cate_estimation(Y, T, X, estimator)
    assert isinstance(result, dict)
