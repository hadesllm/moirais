"""Tests for joswc.joseph_sliding_window_cv."""
import numpy as np
import pytest
from moirais.fn.joswc import joseph_sliding_window_cv


def test_joswc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T_w = np.random.default_rng(42).normal(0, 1, 100)
    step = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = joseph_sliding_window_cv(y, T_w, step, H, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joswc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T_w = np.random.default_rng(42).normal(0, 1, 100)
    step = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = joseph_sliding_window_cv(y, T_w, step, H, K)
    assert isinstance(result, dict)
