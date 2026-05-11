"""Tests for crsfst.crs_forest."""
import numpy as np
import pytest
from morie.fn.crsfst import crs_forest


def test_crsfst_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = crs_forest(time, event, D, X, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crsfst_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = crs_forest(time, event, D, X, K)
    assert isinstance(result, dict)
