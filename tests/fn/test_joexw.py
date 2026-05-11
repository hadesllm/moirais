"""Tests for joexw.joseph_expanding_window_cv."""
import numpy as np
import pytest
from morie.fn.joexw import joseph_expanding_window_cv


def test_joexw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T0 = np.random.default_rng(42).normal(0, 1, 100)
    step = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = joseph_expanding_window_cv(y, T0, step, H, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joexw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    T0 = np.random.default_rng(42).normal(0, 1, 100)
    step = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = joseph_expanding_window_cv(y, T0, step, H, K)
    assert isinstance(result, dict)
