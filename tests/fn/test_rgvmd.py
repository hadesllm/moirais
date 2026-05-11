"""Tests for rgvmd.rangayyan_vmd."""
import numpy as np
import pytest
from morie.fn.rgvmd import rangayyan_vmd


def test_rgvmd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    tau = 0.1
    init = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_vmd(x, K, alpha, tau, init, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgvmd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    tau = 0.1
    init = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_vmd(x, K, alpha, tau, init, tol)
    assert isinstance(result, dict)
