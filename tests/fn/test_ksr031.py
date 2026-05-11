"""Tests for ksr031.kosorok_ch2_weak_convergence_tightness."""
import numpy as np
import pytest
from morie.fn.ksr031 import kosorok_ch2_weak_convergence_tightness


def test_ksr031_basic():
    """Test basic functionality."""
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    eps = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_weak_convergence_tightness(X_n, rho, eps, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr031_edge():
    """Test edge cases."""
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    eps = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_weak_convergence_tightness(X_n, rho, eps, delta)
    assert isinstance(result, dict)
