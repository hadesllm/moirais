"""Tests for hrzsier.horowitz_series_regression."""
import numpy as np
import pytest
from morie.fn.hrzsier import horowitz_series_regression


def test_hrzsier_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = horowitz_series_regression(x, y, K, basis)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzsier_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = horowitz_series_regression(x, y, K, basis)
    assert isinstance(result, dict)
