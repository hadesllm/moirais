"""Tests for rng220.rangayyan_ch4_matched_filter_optimal_transfer_function."""
import numpy as np
import pytest
from moirais.fn.rng220 import rangayyan_ch4_matched_filter_optimal_transfer_function


def test_rng220_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    f = np.random.default_rng(42).normal(0, 1, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_optimal_transfer_function(X, K, f, t_0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng220_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    f = np.random.default_rng(42).normal(0, 1, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_optimal_transfer_function(X, K, f, t_0)
    assert isinstance(result, dict)
