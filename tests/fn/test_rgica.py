"""Tests for rgica.rangayyan_fastica."""
import numpy as np
import pytest
from moirais.fn.rgica import rangayyan_fastica


def test_rgica_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    nonlin = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_fastica(X, n_components, nonlin, max_iter, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgica_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    nonlin = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_fastica(X, n_components, nonlin, max_iter, tol)
    assert isinstance(result, dict)
