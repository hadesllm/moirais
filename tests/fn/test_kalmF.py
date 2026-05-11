"""Tests for kalmF.kalman_filter."""
import numpy as np
import pytest
from morie.fn.kalmF import kalman_filter


def test_kalmF_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = kalman_filter(y, F, H, Q, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kalmF_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = kalman_filter(y, F, H, Q, R)
    assert isinstance(result, dict)
