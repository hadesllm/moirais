"""Tests for kalmS.kalman_smoother."""
import numpy as np
import pytest
from moirais.fn.kalmS import kalman_smoother


def test_kalmS_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = kalman_smoother(y, F, H, Q, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kalmS_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = kalman_smoother(y, F, H, Q, R)
    assert isinstance(result, dict)
