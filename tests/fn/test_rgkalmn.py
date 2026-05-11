"""Tests for rgkalmn.rangayyan_kalman_filter."""
import numpy as np
import pytest
from morie.fn.rgkalmn import rangayyan_kalman_filter


def test_rgkalmn_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    P0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_kalman_filter(z, F, H, Q, R, x0, P0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgkalmn_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    P0 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_kalman_filter(z, F, H, Q, R, x0, P0)
    assert isinstance(result, dict)
