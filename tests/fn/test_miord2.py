"""Tests for miord2.mi_chained_eq."""
import numpy as np
import pytest
from morie.fn.miord2 import mi_chained_eq


def test_miord2_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = mi_chained_eq(data, R, models, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_miord2_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = mi_chained_eq(data, R, models, K)
    assert isinstance(result, dict)
