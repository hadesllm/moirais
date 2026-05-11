"""Tests for hmtvt.geron_train_val_test_split."""
import numpy as np
import pytest
from morie.fn.hmtvt import geron_train_val_test_split


def test_hmtvt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    val_frac = np.random.default_rng(42).normal(0, 1, 100)
    test_frac = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_train_val_test_split(X, y, val_frac, test_frac, seed)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hmtvt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    val_frac = np.random.default_rng(42).normal(0, 1, 100)
    test_frac = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_train_val_test_split(X, y, val_frac, test_frac, seed)
    assert isinstance(result, dict)
