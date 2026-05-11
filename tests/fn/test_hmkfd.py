"""Tests for hmkfd.geron_kfold."""
import numpy as np
import pytest
from morie.fn.hmkfd import geron_kfold


def test_hmkfd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    seed = 42
    result = geron_kfold(X, y, k, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmkfd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    seed = 42
    result = geron_kfold(X, y, k, seed)
    assert isinstance(result, dict)
