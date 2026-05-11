"""Tests for hmins.geron_instance_based."""
import numpy as np
import pytest
from morie.fn.hmins import geron_instance_based


def test_hmins_basic():
    """Test basic functionality."""
    X_train = np.random.default_rng(42).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    x_query = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_instance_based(X_train, y_train, x_query, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmins_edge():
    """Test edge cases."""
    X_train = np.random.default_rng(42).normal(0, 1, 100)
    y_train = np.random.default_rng(43).normal(0, 1, 100)
    x_query = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_instance_based(X_train, y_train, x_query, k)
    assert isinstance(result, dict)
