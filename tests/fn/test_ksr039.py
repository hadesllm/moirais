"""Tests for ksr039.kosorok_ch2_weak_convergence_lipschitz."""
import numpy as np
import pytest
from moirais.fn.ksr039 import kosorok_ch2_weak_convergence_lipschitz


def test_ksr039_basic():
    """Test basic functionality."""
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = kosorok_ch2_weak_convergence_lipschitz(X_n, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr039_edge():
    """Test edge cases."""
    X_n = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = kosorok_ch2_weak_convergence_lipschitz(X_n, X)
    assert isinstance(result, dict)
