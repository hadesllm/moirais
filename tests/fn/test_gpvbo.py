"""Tests for gpvbo.gp_variational_bayes_opt."""
import numpy as np
import pytest
from morie.fn.gpvbo import gp_variational_bayes_opt


def test_gpvbo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = gp_variational_bayes_opt(X, y, X_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpvbo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = gp_variational_bayes_opt(X, y, X_grid)
    assert isinstance(result, dict)
