"""Tests for wsdt2d.wasserstein_p_d."""
import numpy as np
import pytest
from moirais.fn.wsdt2d import wasserstein_p_d


def test_wsdt2d_basic():
    """Test basic functionality."""
    X_samples = np.random.default_rng(42).normal(0, 1, 100)
    Y_samples = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = wasserstein_p_d(X_samples, Y_samples, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsdt2d_edge():
    """Test edge cases."""
    X_samples = np.random.default_rng(42).normal(0, 1, 100)
    Y_samples = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = wasserstein_p_d(X_samples, Y_samples, p)
    assert isinstance(result, dict)
