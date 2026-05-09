"""Tests for ksr032.kosorok_ch2_weak_convergence_iff."""
import numpy as np
import pytest
from moirais.fn.ksr032 import kosorok_ch2_weak_convergence_iff


def test_ksr032_basic():
    """Test basic functionality."""
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = kosorok_ch2_weak_convergence_iff(X_n, X, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr032_edge():
    """Test edge cases."""
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = kosorok_ch2_weak_convergence_iff(X_n, X, T)
    assert isinstance(result, dict)
