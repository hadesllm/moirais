"""Tests for hmsem.geron_semisupervised."""
import numpy as np
import pytest
from moirais.fn.hmsem import geron_semisupervised


def test_hmsem_basic():
    """Test basic functionality."""
    X_l = np.random.default_rng(42).normal(0, 1, 100)
    y_l = np.random.default_rng(42).normal(0, 1, 100)
    X_u = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = geron_semisupervised(X_l, y_l, X_u, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsem_edge():
    """Test edge cases."""
    X_l = np.random.default_rng(42).normal(0, 1, 100)
    y_l = np.random.default_rng(42).normal(0, 1, 100)
    X_u = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = geron_semisupervised(X_l, y_l, X_u, alpha)
    assert isinstance(result, dict)
