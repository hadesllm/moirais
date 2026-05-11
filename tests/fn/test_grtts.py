"""Tests for grtts.geron_train_test_split."""
import numpy as np
import pytest
from morie.fn.grtts import geron_train_test_split


def test_grtts_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    test_size = 100
    seed = 42
    result = geron_train_test_split(X, y, test_size, seed)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_grtts_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    test_size = 100
    seed = 42
    result = geron_train_test_split(X, y, test_size, seed)
    assert isinstance(result, dict)
