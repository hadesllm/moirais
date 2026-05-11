"""Tests for grkfd.geron_kfold_cv."""
import numpy as np
import pytest
from morie.fn.grkfd import geron_kfold_cv


def test_grkfd_basic():
    """Test basic functionality."""
    n = 100
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    shuffle = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_kfold_cv(n, K, shuffle, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grkfd_edge():
    """Test edge cases."""
    n = 100
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    shuffle = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_kfold_cv(n, K, shuffle, seed)
    assert isinstance(result, dict)
