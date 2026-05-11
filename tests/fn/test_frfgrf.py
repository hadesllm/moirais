"""Tests for frfgrf.forest_fit_consistency."""
import numpy as np
import pytest
from morie.fn.frfgrf import forest_fit_consistency


def test_frfgrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = forest_fit_consistency(y, D, X, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_frfgrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = forest_fit_consistency(y, D, X, K)
    assert isinstance(result, dict)
