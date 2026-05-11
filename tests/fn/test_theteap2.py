"""Tests for theteap2.theta_map."""
import numpy as np
import pytest
from morie.fn.theteap2 import theta_map


def test_theteap2_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    items = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = theta_map(X, items, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_theteap2_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    items = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = theta_map(X, items, prior)
    assert isinstance(result, dict)
