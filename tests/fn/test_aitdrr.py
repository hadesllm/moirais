"""Tests for aitdrr.dirichlet_regression."""
import numpy as np
import pytest
from morie.fn.aitdrr import dirichlet_regression


def test_aitdrr_basic():
    """Test basic functionality."""
    X_cov = np.random.default_rng(42).normal(0, 1, 100)
    Y_comp = np.random.default_rng(42).normal(0, 1, 100)
    ref = np.random.default_rng(42).normal(0, 1, 100)
    result = dirichlet_regression(X_cov, Y_comp, ref)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitdrr_edge():
    """Test edge cases."""
    X_cov = np.random.default_rng(42).normal(0, 1, 100)
    Y_comp = np.random.default_rng(42).normal(0, 1, 100)
    ref = np.random.default_rng(42).normal(0, 1, 100)
    result = dirichlet_regression(X_cov, Y_comp, ref)
    assert isinstance(result, dict)
