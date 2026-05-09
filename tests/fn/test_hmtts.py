"""Tests for hmtts.geron_train_test_split."""
import numpy as np
import pytest
from moirais.fn.hmtts import geron_train_test_split


def test_hmtts_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    test_size = 100
    seed = 42
    result = geron_train_test_split(X, y, test_size, seed)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hmtts_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    test_size = 100
    seed = 42
    result = geron_train_test_split(X, y, test_size, seed)
    assert isinstance(result, dict)
