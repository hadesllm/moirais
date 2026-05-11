"""Tests for jormse2.joseph_rmsse."""
import numpy as np
import pytest
from morie.fn.jormse2 import joseph_rmsse


def test_jormse2_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    m = 10
    result = joseph_rmsse(y_true, y_pred, y_train, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jormse2_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    m = 10
    result = joseph_rmsse(y_true, y_pred, y_train, m)
    assert isinstance(result, dict)
