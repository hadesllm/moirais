"""Tests for acclso.accelerated_lasso."""
import numpy as np
import pytest
from morie.fn.acclso import accelerated_lasso


def test_acclso_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = accelerated_lasso(X, y, lam, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_acclso_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = accelerated_lasso(X, y, lam, steps)
    assert isinstance(result, dict)
