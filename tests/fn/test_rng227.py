"""Tests for rng227.rangayyan_ch4_matched_filter_optimal_H_eeg."""
import numpy as np
import pytest
from moirais.fn.rng227 import rangayyan_ch4_matched_filter_optimal_H_eeg


def test_rng227_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    f = np.random.default_rng(42).normal(0, 1, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_optimal_H_eeg(X, K, f, t_0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng227_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    f = np.random.default_rng(42).normal(0, 1, 100)
    t_0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_optimal_H_eeg(X, K, f, t_0)
    assert isinstance(result, dict)
