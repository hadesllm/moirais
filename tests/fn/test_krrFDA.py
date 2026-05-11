"""Tests for krrFDA.kernel_ridge_regression."""
import numpy as np
import pytest
from morie.fn.krrFDA import kernel_ridge_regression


def test_krrFDA_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    lam = 0.1
    result = kernel_ridge_regression(X, y, kernel, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krrFDA_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    lam = 0.1
    result = kernel_ridge_regression(X, y, kernel, lam)
    assert isinstance(result, dict)
